SELECT C.country_name
FROM countries AS C, regions AS R 
WHERE C.region_id = R.region_id and R.region_name = "Europe";