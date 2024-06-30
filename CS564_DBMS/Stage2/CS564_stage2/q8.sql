SELECT manager_id, salary 
FROM employees 
WHERE salary = (SELECT min(salary) FROM employees);