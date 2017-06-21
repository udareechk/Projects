<?php
	require 'Credentials.php';	

	
	$CourseID = $_POST['course'];
	$GPA = $_POST['gpa'];
	$ID = $_COOKIE['user'];

	$connection = mysqli_connect($servername, $username, $password, $database);

	mysqli_query($connection, "INSERT INTO GPA VALUES ('$ID','$CourseID','$GPA')");

	header("Location: Update.php");
	exit();
	mysqli_close($connection);
?>
