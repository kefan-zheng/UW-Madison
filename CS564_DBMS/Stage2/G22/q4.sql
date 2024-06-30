SELECT count(E.employee_id) AS number_of_employees_in_Shipping
FROM departments AS D, employees AS E
WHERE D.department_id = E.department_id and D.department_name = "Shipping";
