
<?php
	require 'Credentials.php';	
	
	$connection = mysqli_connect($servername, $username, $password, $database);

	$Password = $_POST['password'];
	$Email = $_POST['email'];

	$Query = mysqli_query($connection, "SELECT Password, ID FROM ACCOUNT WHERE Email = '$Email'");
	$Account = mysqli_fetch_array($Query);

	if ($Password == $Account['Password']) {
		$cookie_ID = "user";
		$cookie_val = $Account['ID'];
		setcookie($cookie_ID, $cookie_val, time() + (86400 * 30), "/");
		header("Location: Profile.php");
		exit();

	} else {
		header("Location: index.html");
	}

	mysqli_close($connection);
?>
