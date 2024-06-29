SELECT E1.first_name, E2.first_name 
FROM employees E1, employees E2 
WHERE E1.employee_id != E2.employee_id
AND E1.manager_id = E2.manager_id 
AND E1.department_id = E2.department_id 
AND E1.salary >= E2.salary 
AND E1.salary > 10000 
AND E2.salary > 10000;
