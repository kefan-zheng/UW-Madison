SELECT D.department_name, avg(J.max_salary) AS average_max_salary 
FROM employees AS E, departments AS D, jobs AS J 
WHERE E.department_id = D.department_id AND E.job_id = J.job_id 
GROUP BY E.department_id HAVING avg(J.max_salary) > 8000;