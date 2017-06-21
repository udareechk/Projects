<?php 

	require 'Credentials.php';	
	$connection = mysqli_connect($servername, $username, $password, $database);

	$ID = $_COOKIE['user'];
	$FName = $_POST['firstName'];
	$LName = $_POST['lastName'];
	$Initials = $_POST['initials'];
	$Eno = $_POST['eno'];
	$Gender = $_POST['gender'];
	$UniID = $_POST['university'];
	$FacultyID = $_POST['faculty'];
	$Batch = $_POST['batch'];

	$Email = $_POST['email'];
	$Password1 = $_POST['password'];
	$Password2 = $_POST['password2'];

	if ($Password1 == $Password2){
		mysqli_query($connection, "INSERT INTO ACCOUNT (Email, Password, AccountType) VALUES ('$Email', '$Password1', 1)");
		$GetID = mysqli_query($connection, "SELECT MAX(ID) FROM ACCOUNT");
		$IDArr = mysqli_fetch_array($GetID);
		$ID = $IDArr['MAX(ID)'];
		mysqli_query($connection, "INSERT INTO STUDENT (ID, FName, Initials, LName, Eno, Gender, UniID, FacultyID, Batch) VALUES ($ID, '$FName', '$Initials', '$LName', '$Eno', $Gender, $UniID, $FacultyID, $Batch)");

		$cookie_ID = "user";
		$cookie_val = $ID;
		setcookie($cookie_ID, $cookie_val, time() + (86400 * 30), "/");
		header("Location: Profile.php");
	}
 	else {
 		echo "Input Password does not Match!";
 	}
	mysqli_close($connection);

?>