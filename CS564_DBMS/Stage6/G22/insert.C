#include "catalog.h"
#include "query.h"


/*
 * Inserts a record into the specified relation.
 *
 * Returns:
 * 	OK on success
 * 	an error code otherwise
 */

const Status QU_Insert(const string & relation, 
	const int attrCnt, 
	const attrInfo attrList[])
{
	cout << "Doing QU_Insert" << endl;

	Status status = OK;

	// get record len
	int reclen = 0;
	AttrDesc attrListDesc[attrCnt];
	for(int i=0;i<attrCnt;i++){
		status = attrCat->getInfo(relation, attrList[i].attrName, attrListDesc[i]);
		if(status != OK) return status;
		// update reclen
		reclen += attrListDesc[i].attrLen;
	}
	if(reclen == 0) return INVALIDRECLEN;


	// traversal attr
	char insertData[reclen];
	Record insertRec;
	insertRec.data = (void*) insertData;
	insertRec.length = reclen;
	for(int i=0;i<attrCnt;i++){
		// copy attr
		if(attrListDesc[i].attrType == INTEGER){
			int value = atoi((char*)attrList[i].attrValue);
			memcpy(insertData + attrListDesc[i].attrOffset, &value, attrListDesc[i].attrLen);

		}else if(attrListDesc[i].attrType == FLOAT){
			float value = atof((char*)attrList[i].attrValue);
			memcpy(insertData + attrListDesc[i].attrOffset, &value, attrListDesc[i].attrLen);

		}else{
			memcpy(insertData + attrListDesc[i].attrOffset, (char*)attrList[i].attrValue, attrListDesc[i].attrLen);
		}
	}

	// create insert scan 
	InsertFileScan table = InsertFileScan(relation, status);
	if(status != OK) return status;

	// insert record
	RID tmpRid;
	status = table.insertRecord(insertRec, tmpRid);
	if(status != OK) return status;

	// part 6
	return OK;

}

