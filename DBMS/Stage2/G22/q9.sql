SELECT D.department_name
FROM employees E, departments D 
WHERE E.department_id = D.department_id 
AND E.salary = (SELECT max(salary) FROM employees); 