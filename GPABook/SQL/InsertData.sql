TRUNCATE TABLE UNIVERSITY;
TRUNCATE TABLE FACULTY;
TRUNCATE TABLE DEPARTMENT;
TRUNCATE TABLE COURSES;

INSERT INTO UNIVERSITY (UniName) VALUES ('University of Peradeniya');

INSERT INTO FACULTY(FacName, UniID) VALUES ('Faculty of Engineering',1);

INSERT INTO DEPARTMENT (DName, FacultyID) VALUES 
('Department of Computer Engineering',1),
('Department of Electrical and Electronic Engineering', 1),
('Department of Engineering Mathematics', 1);

INSERT INTO COURSES (CourseCode, CourseName, DepartmentID, Credits, Elective, PrereqID, Semester, NonGPA) VALUES 
('CO221', 'Digital Design', 1, 3, 0, NULL, 3, 0),
('CO222', 'Programming Methodology', 1, 3, 0, NULL, 3, 0),
('CO223', 'Computer Communication Networks I', 1, 3, 0, NULL, 3, 0),
('CO224', 'Computer Architecture', 1, 3, 0, NULL, 4, 0),
('CO225', 'Software Construction', 1, 3, 0, NULL, 4, 0),
('CO226', 'Database Systems', 1, 3, 0, NULL, 4, 0),
('EE282', 'Network Analysis', 2, 3, 0, NULL, 3, 0),
('EE285', 'Electronics 1', 2, 3, 0, NULL, 4, 0),
('EM201', 'Mathematics III', 3, 3, 0, NULL, 3, 0),
('EM202', 'Mathematics IV', 3, 3, 0, NULL, 4, 0),
('EM313', 'Descrete Mathematics', 3, 3, 0, NULL, 3, 0),
('EM314', 'Numerical Methods', 3, 3, 0, NULL, 4, 0);
