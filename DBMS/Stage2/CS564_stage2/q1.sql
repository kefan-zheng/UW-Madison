SELECT DISTINCT job_title
FROM jobs
WHERE max_salary >= 5000 AND min_salary <= 4000;