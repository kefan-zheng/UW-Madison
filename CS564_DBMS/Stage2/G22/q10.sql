SELECT E.employee_id
FROM employees E 
WHERE E.employee_id NOT IN
(SELECT DISTINCT employee_id FROM dependents);