#include "catalog.h"
#include "query.h"


// forward declaration
const Status ScanSelect(const string & result, 
			const int projCnt, 
			const AttrDesc projNames[],
			const AttrDesc *attrDesc, 
			const Operator op, 
			const char *filter,
			const int reclen);

/*
 * Selects records from the specified relation.
 *
 * Returns:
 * 	OK on success
 * 	an error code otherwise
 */

const Status QU_Select(const string & result, 
		       const int projCnt, 
		       const attrInfo projNames[],
		       const attrInfo *attr, 
		       const Operator op, 
		       const char *attrValue)
{
   // Qu_Select sets up things and then calls ScanSelect to do the actual work
   	cout << "Doing QU_Select " << endl;

	Status status = OK;

	// set
   	int reclen = 0;
   	AttrDesc projNamesDesc[projCnt];

	// traversal project attr
	for(int i=0;i<projCnt;i++){
		string relation = projNames[i].relName;
		string attrName = projNames[i].attrName;
		// get project name desc
		status = attrCat->getInfo(relation, attrName, projNamesDesc[i]);
		reclen += projNamesDesc[i].attrLen;
		if(status != OK) return status;
	}
	
	// get target attr desc
	AttrDesc attrDesc;
	if(attr){
		status = attrCat->getInfo(string(attr->relName), string(attr->attrName), attrDesc);
		if(status != OK) return status;
	}
	
	// scan select
	status = ScanSelect(result, projCnt, projNamesDesc, &attrDesc, op, attrValue, reclen);
	if(status != OK) return status;

	return OK;
}


const Status ScanSelect(const string & result, 
#include "stdio.h"
#include "stdlib.h"
			const int projCnt, 
			const AttrDesc projNames[],
			const AttrDesc *attrDesc, 
			const Operator op, 
			const char *filter,
			const int reclen)
{
    cout << "Doing HeapFileScan Selection using ScanSelect()" << endl;

	Status status = OK;

	// init output
	InsertFileScan outputTable = InsertFileScan(result, status);
	if(status != OK) return status;

	// init input
	HeapFileScan inputTable = HeapFileScan(string(projNames[0].relName), status);
	if(status != OK) return status;

	// set scan
	int filter_i;
	float filter_f;
	if(attrDesc && filter){
		Datatype type = (Datatype) attrDesc->attrType;
		if(type == INTEGER){
			filter_i = atoi(filter);
			status = inputTable.startScan(attrDesc->attrOffset, attrDesc->attrLen, type, (char*)&filter_i, op);
		}else if(type == FLOAT){
			filter_f = atof(filter);
			status = inputTable.startScan(attrDesc->attrOffset, attrDesc->attrLen, type, (char*)&filter_f, op);
		}else{
			status = inputTable.startScan(attrDesc->attrOffset, attrDesc->attrLen, type, (char*)filter, op);
		}
		if(status != OK) return status;
	}else{
		status = inputTable.startScan(0, 0, STRING, NULL, op);
		if(status != OK) return status;
	}
	
	// scan
	RID tmpRid;
	while(inputTable.scanNext(tmpRid) == OK){
		// get record
		Record rec;
		status = inputTable.getRecord(rec);
		if(status != OK) return status;
		// construct output record
		char outputData[reclen];
		Record outputRec;
		outputRec.data = (void*) outputData;
		outputRec.length = reclen;
		// project attr
		int outputOffset = 0;
		for(int i=0;i<projCnt;i++){
			memcpy(outputData + outputOffset, (char*)rec.data + projNames[i].attrOffset, projNames[i].attrLen);
			outputOffset += projNames[i].attrLen;
		}
		// insert record into output
		status = outputTable.insertRecord(outputRec, tmpRid);
		if(status != OK) return status;
	}

	// end scan
	status = inputTable.endScan();
	if(status != OK) return status;

	return OK;
}
