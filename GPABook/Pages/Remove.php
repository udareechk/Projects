<?php
	require 'Credentials.php';	
	
	$CourseID = $_POST['course'];
	$ID = $_COOKIE['user'];

	$connection = mysqli_connect($servername, $username, $password, $database);

	mysqli_query($connection, "DELETE FROM GPA WHERE StudentID = $ID AND CourseID = $CourseID");

	header("Location: Update.php");
	exit();
	mysqli_close($connection);
?>
