<!DOCTYPE html>

<?php
	require 'Credentials.php';	
?>

<html>

<head>
	<title>Profile</title>

	<style type="text/css">
			body{
				margin:0;
				background-color: #ccffef;
				min-width: auto;
				min-height: auto;

			}

		#TopHeader{
				width: 100%;
				height: 80px;
				background-color:#19194d;
				box-shadow: 0px 2px 10px #000033;

			}

			#aboutUs{
				padding-left: 10px;
			}

			#logout {
				margin-left: 100px;
				position: absolute;
				width: 100px;
				margin-top: -20px;
				opacity: 0.8;
			}

		#TopHeader>h1{
				position: absolute;
				left:7% ;
				color: white;
				font-size: 50px;
				font-family: arial;
				margin: 0px auto;
				padding: 0.5%;
			}	

			#others{
				position:absolute;
				text-align: center;
				color: white;
				top: 4%;
				left:80%;

			}

			.form{
				background-color: white;
				position: relative;
				margin: auto;
				top:10px;
    			width: 900px;
				height:1200px ;
			}

			#profileImage{
				background-color: white;
				position: relative;
				margin-left: 20px;
				top:20px;
    			width: 200px;
    			border: solid;
    			border-radius: 10px;
				height: 200px ;
			}

			#upperPanel{
				background-color:#fff5cc ;
				position: relative;
    			width: 100%;
				height: 250px ;
			}

			#simpleData{
				position: absolute;
				position: relative;
				margin-left: 230px;
				margin-top: -180px;
    			width: 655px;
				height: 200px ;
			}

			.leftPanel{
				position: relative;
				margin-top: -20px;
    			width: 150%;
    			border: none;
				height: 150%;
			}

			.rightPanel{
				position: absolute;
				margin-left: 250px;
				margin-top: 10px;
				width: 622px;
				overflow-y: hidden;
				height: 1000px;

			}	

			.buttons {
				margin-left: 100px;
			}

			
</style>
</head>


<body>

<div id ="TopHeader">
<h1><span font-family: courier>GPA</span> Book</h1>
</div>

<div id = "others">
	<a id="aboutGpaBook" href=""><span style="color:white">About us </span></a>
	<a id="aboutUs" href=""><span id="logout" style="color:white"> 

		<form action = "Index.html">
		<button>Log Out</button>
		</form>
	</span></a>
</div>

<div class ="form">

	<div id = "upperPanel">
		<div id = "profileImage">
			<div><img src='../Images/ProfilePictures/0.jpg' style='width:200px;height:200px'></div>
		<div>
		</div>
		<div id="simpleData">
		<div>
			<?php

			$connection = mysqli_connect($servername, $username, $password, $database);
			$ID = $_COOKIE['user'];

			$GPAData = mysqli_query($connection, "SELECT GPA FROM GPA WHERE StudentID = $ID");
			
			$MyGPA = 0;
			$Count = 0;

			while ($Data = mysqli_fetch_array($GPAData)){
				$MyGPA = $MyGPA + $Data['GPA'];
				$Count = $Count + 1;
			}

			if ($Count == 0){
				echo "<br><h1>Input your Results to calculate your GPA</h1>";
			}
			else {
				$MyGPA = $MyGPA / $Count;
				echo "<br><h1>GPA: <u>".number_format((float)$MyGPA, 2, '.', '')."</u> out of $Count Course(s)</h1>";
			}
			mysqli_close($connection);
		?>
		</div>
		
		<div>
			<form action = "Update.php">
			<button class="buttons">Update Data</button>
			</form>
		</div>
		<br>
		<div>
			<form action = "Stats.php">
			<button class="buttons">View My Stats</button>
			</form>
		</div>

	</div>
	<div class ="rightPanel">
		<div>
			
		</div>
		

	</div>	

	<div class = "leftPanel">
		<?php

		$connection = mysqli_connect($servername, $username, $password, $database);
		$ID = $_COOKIE['user'];
		$Profile = mysqli_query($connection, "SELECT FName, LName, Initials, ENo, Batch, UniName, FacName FROM STUDENT, UNIVERSITY, FACULTY WHERE StudentID = $ID AND UNIVERSITY.UniID = STUDENT.UniID AND FACULTY.FacultyID = STUDENT.FacultyID");

		$Data = mysqli_fetch_array($Profile);

		echo " <div id='userData'><h1>$Data[FName] $Data[LName]</h1>
				
					<h2>$Data[Initials] $Data[LName]</h2>
					<h3>$Data[ENo]</h3>
					<h3>$Data[FacName]</h3>
					<h3>$Data[UniName]</h3>
				</div>";
		?>
	</div>


</body>
</html>