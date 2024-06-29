#include "catalog.h"
#include "query.h"


/*
 * Deletes records from a specified relation.
 *
 * Returns:
 * 	OK on success
 * 	an error code otherwise
 */

const Status QU_Delete(const string & relation, 
		       const string & attrName, 
		       const Operator op,
		       const Datatype type, 
		       const char *attrValue)
{
	cout << "Doing QU_Delete " << endl;

	Status status = OK;
	
	// create scan
	HeapFileScan inputTable = HeapFileScan(relation, status);
	if(status != OK) return status;
	
	// set scan
	int filter_i;
	float filter_f;
	if(!relation.empty() && !attrName.empty()){
		// get attribute desc
		AttrDesc attrDesc;
		status = attrCat->getInfo(relation, attrName, attrDesc);
		if(status != OK) return status;

		// set scan
		if(type == INTEGER){
			filter_i = atoi(attrValue);
			status = inputTable.startScan(attrDesc.attrOffset, attrDesc.attrLen, type, (char*)&filter_i, op);
		}else if(type == FLOAT){
			filter_f = atof(attrValue);
			status = inputTable.startScan(attrDesc.attrOffset, attrDesc.attrLen, type, (char*)&filter_f, op);
		}else{
			status = inputTable.startScan(attrDesc.attrOffset, attrDesc.attrLen, type, (char*)attrValue, op);
		}
	}else{
		status = inputTable.startScan(0, 0, type, NULL, op);
	}
	if(status != OK) return status;

	// start scan
	RID tmpRid;
	while((status = inputTable.scanNext(tmpRid)) == OK){
		status = inputTable.deleteRecord();
		if(status != OK) return status;
	}

	// end scan
	status = inputTable.endScan();
	if(status != OK) return status;

	// part 6
	return OK;
}


