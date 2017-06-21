<!DOCTYPE html>

<?php
	require 'Credentials.php';	
?>

<html>
<head>
	<title>Rank</title>

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
		table {
		    font-family: arial, sans-serif;
		    border-collapse: collapse;
		    width: 100%;
		    width: 90%;
		    margin-left: 5%;
		}

		td, th {
		    border: 1px solid #dddddd;
		    text-align: center;
		    padding: 8px;
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
<?php

	$CourseID = $_POST['course'];
	
	$connection = mysqli_connect($servername, $username, $password, $database);

	$CourseData = mysqli_query($connection, "SELECT CourseCode, CourseName FROM COURSES WHERE CourseID = $CourseID");
	$Courses = mysqli_query($connection, "SELECT FName, LName, Eno, GPA FROM STUDENT, GPA WHERE STUDENT.StudentID = GPA.StudentID AND GPA.CourseID = $CourseID ORDER BY GPA DESC");

	$CourseDetails = mysqli_fetch_array($CourseData);

	echo "<h1><u>$CourseDetails[CourseCode] - $CourseDetails[CourseName] Ranks</u></h1>";
	
	echo "<table>";

	echo "<tr>
			<th>Rank</th>
			<th>Name</th>
			<th>E-No</th>
			<th>GPA</th>
		</tr>";

	$count = 1;
	while ($Data = mysqli_fetch_array($Courses)) {
		echo"<tr>
				<td>$count</td>
				<td>$Data[FName] $Data[LName]</td>
				<td>$Data[Eno]</td>
				<td>$Data[GPA]</td>
			</tr>";
			$count = $count+1;
	}
	
	echo "</table>";
	mysqli_close($connection);
?>

	<br>
	<div>
		<form action = "Stats.php">
		<button>Another Course</button>
		</form>
	</div>
	<br>
	<div>
		<form action = "Profile.php">
		<button>View Profile</button>
		</form>
	</div>

</div>

</body>
</html>