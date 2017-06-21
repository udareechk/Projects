<!DOCTYPE html>

<?php
	require 'Credentials.php';	
?>

<html>
<head>
	<title>Statistics</title>

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

<div class="form">
	<h1>Rank of the Batch</h1>
	<form action="Rank.php" method="POST">
	<div>
		<h2><u>Rank by Each Course</u></h2>

		<h4>Select your course code from the list</h4>

		<?php

			$connection = mysqli_connect($servername, $username, $password, $database);
			$Courses = mysqli_query($connection, "SELECT CourseID, CourseName, CourseCode FROM COURSES ORDER BY CourseID");

			echo "<select name='course'>";
			while ($Course = mysqli_fetch_array($Courses)) {
				echo "<option value='$Course[CourseID]'>$Course[CourseCode] - $Course[CourseName]</option>";
			}
			echo "</select>";
			mysqli_close($connection);
		?>

	</div>
	
	<br>
	<input type="submit" value="View"/>
	</form>
</div>

</body>
</html>