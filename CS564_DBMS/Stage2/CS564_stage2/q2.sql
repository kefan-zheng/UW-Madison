SELECT D.department_name, count(E.employee_id) AS number_of_employees
FROM departments AS D
LEFT JOIN employees AS E
ON D.department_id = E.department_id
GROUP BY D.department_id
ORDER BY number_of_employees DESC;