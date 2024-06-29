#include <memory.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
#include <iostream>
#include <stdio.h>
#include "page.h"
#include "buf.h"

#define ASSERT(c)  { if (!(c)) { \
		       cerr << "At line " << __LINE__ << ":" << endl << "  "; \
                       cerr << "This condition should hold: " #c << endl; \
                       exit(1); \
		     } \
                   }

//----------------------------------------
// Constructor of the class BufMgr      //
//----------------------------------------

BufMgr::BufMgr(const int bufs)
{
    numBufs = bufs;

    bufTable = new BufDesc[bufs];
    memset(bufTable, 0, bufs * sizeof(BufDesc));
    for (int i = 0; i < bufs; i++) 
    {
        bufTable[i].frameNo = i;
        bufTable[i].valid = false;
    }

    bufPool = new Page[bufs];
    memset(bufPool, 0, bufs * sizeof(Page));

    int htsize = ((((int) (bufs * 1.2))*2)/2)+1;
    hashTable = new BufHashTbl (htsize);  // allocate the buffer hash table

    clockHand = bufs - 1;
}


BufMgr::~BufMgr() {

    // flush out all unwritten pages
    for (int i = 0; i < numBufs; i++) 
    {
        BufDesc* tmpbuf = &bufTable[i];
        if (tmpbuf->valid == true && tmpbuf->dirty == true) {

#ifdef DEBUGBUF
            cout << "flushing page " << tmpbuf->pageNo
                 << " from frame " << i << endl;
#endif

            tmpbuf->file->writePage(tmpbuf->pageNo, &(bufPool[i]));
        }
    }

    delete [] bufTable;
    delete [] bufPool;
}


const Status BufMgr::allocBuf(int & frame) 
{
    // check whether all buffer frames are pinned
    int checkFrameNum = 0;
    while(checkFrameNum < 2 * numBufs){
        // advance clock pointer
        advanceClock();
        // get frame description
        frame = clockHand;
        BufDesc* currentFrameDesc = &(bufTable[frame]);
        // check valid
        if(currentFrameDesc->valid){
            // check refBit
            if(!currentFrameDesc->refbit){
                // check whether pinned
                if(currentFrameDesc->pinCnt <= 0){
                    // check whether page dirty
                    if(currentFrameDesc->dirty){
                        // flush page to disk
                        if(currentFrameDesc->file->writePage(currentFrameDesc->pageNo, &(bufPool[frame])) != OK){
                            return UNIXERR;
                        }
                    }
                    // update hash table (if check remove error, cant pass the test)
                    hashTable->remove(currentFrameDesc->file, currentFrameDesc->pageNo);
                    // return frame number
                    currentFrameDesc->frameNo = frame;
                    return OK;
                }
            }else{
                // if refBit true
                currentFrameDesc->refbit = false;
            }
        }else{
            // not valid, return frame number directly
            currentFrameDesc->frameNo = frame;
            return OK;
        }

        checkFrameNum++;
    }

    return BUFFEREXCEEDED;
}

	
const Status BufMgr::readPage(File* file, const int PageNo, Page*& page)
{
    Status globalStatus;
    // check whether page in the buffer pool
    int frameNo = -1;
    Status pageStatus = hashTable->lookup(file, PageNo, frameNo);
    if(pageStatus == OK){ 
        // in the buffer pool
        // update frame description
        BufDesc* frameDesc = &(bufTable[frameNo]);
        frameDesc->refbit = true;
        frameDesc->pinCnt++;
    }else if(pageStatus == HASHNOTFOUND){
        // not in the buffer pool
        // alloc frame
        if((globalStatus = allocBuf(frameNo)) != OK){
            return globalStatus;
        }
        // read page from disk to frame
        if((globalStatus = file->readPage(PageNo, &(bufPool[frameNo]))) != OK){
            return globalStatus;
        }
        // update hash table
        if((globalStatus = hashTable->insert(file, PageNo, frameNo)) != OK){
            return globalStatus;
        }
        // update frame description
        BufDesc* frameDesc = &(bufTable[frameNo]);
        frameDesc->Set(file, PageNo);
    }
    // return page pointer
    page = &(bufPool[frameNo]);

    return OK;
}


const Status BufMgr::unPinPage(File* file, const int PageNo, const bool dirty) 
{
    Status globalStatus;
    // find frame
    int frameNo = -1;
    if((globalStatus = hashTable->lookup(file, PageNo, frameNo)) != OK){
        return globalStatus;
    }
    // update frame description
    BufDesc* frameDesc = &(bufTable[frameNo]);
    if (frameDesc->pinCnt==0){
        return PAGENOTPINNED;
    }
    // reduce pinCnt
    frameDesc->pinCnt--;
    // set dirty
    if(dirty){
        frameDesc->dirty = true;
    }
    return OK;
}

const Status BufMgr::allocPage(File* file, int& pageNo, Page*& page) 
{
    Status globalStatus;
    if((globalStatus = file->allocatePage(pageNo)) != OK){
        return globalStatus;
    }
    // alloc frame
    int frameNo = -1;
    if((globalStatus = allocBuf(frameNo)) != OK){
        return globalStatus;
    }
    // update hash table
    if((globalStatus = hashTable->insert(file, pageNo, frameNo)) != OK){
        return globalStatus;
    }
    // update frame description
    BufDesc* frameDesc = &(bufTable[frameNo]);
    frameDesc->Set(file, pageNo);
    
    page = &(bufPool[frameNo]);

    return OK;
}

const Status BufMgr::disposePage(File* file, const int pageNo) 
{
    // see if it is in the buffer pool
    Status status = OK;
    int frameNo = 0;
    status = hashTable->lookup(file, pageNo, frameNo);
    if (status == OK)
    {
        // clear the page
        bufTable[frameNo].Clear();
    }
    status = hashTable->remove(file, pageNo);

    // deallocate it in the file
    return file->disposePage(pageNo);
}

const Status BufMgr::flushFile(const File* file) 
{
  Status status;

  for (int i = 0; i < numBufs; i++) {
    BufDesc* tmpbuf = &(bufTable[i]);
    if (tmpbuf->valid == true && tmpbuf->file == file) {

      if (tmpbuf->pinCnt > 0)
	  return PAGEPINNED;

      if (tmpbuf->dirty == true) {
        #ifdef DEBUGBUF
            cout << "flushing page " << tmpbuf->pageNo
                    << " from frame " << i << endl;
        #endif
        if ((status = tmpbuf->file->writePage(tmpbuf->pageNo, &(bufPool[i]))) != OK)
            return status;

        tmpbuf->dirty = false;
      }

      hashTable->remove(file,tmpbuf->pageNo);

      tmpbuf->file = NULL;
      tmpbuf->pageNo = -1;
      tmpbuf->valid = false;
    }

    else if (tmpbuf->valid == false && tmpbuf->file == file)
      return BADBUFFER;
  }
  
  return OK;
}


void BufMgr::printSelf(void) 
{
    BufDesc* tmpbuf;
  
    cout << endl << "Print buffer...\n";
    for (int i=0; i<numBufs; i++) {
        tmpbuf = &(bufTable[i]);
        cout << i << "\t" << (char*)(&bufPool[i]) 
             << "\tpinCnt: " << tmpbuf->pinCnt;
    
        if (tmpbuf->valid == true)
            cout << "\tvalid\n";
        cout << endl;
    };
}


