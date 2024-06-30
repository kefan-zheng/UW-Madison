SELECT count(E.employee_id) AS number_of_employees_in_Europe
FROM employees E, departments D, locations L 
WHERE E.department_id = D.department_id 
AND D.location_id = L.location_id 
AND L.country_id 
IN (SELECT C.country_id 
    FROM countries AS C, regions AS R 
    WHERE C.region_id = R.region_id 
    AND R.region_name = "Europe"
);