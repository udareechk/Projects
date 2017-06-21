<!DOCTYPE html>

<?php
	require 'Credentials.php';	
?>

<html>
<head>
	<title>Update Details</title>

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
		table {
		    font-family: arial, sans-serif;
		    border-collapse: collapse;
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
	<div>

	<h1><u>Courses</u></h1>
	
	<?php
		
		$ID = $_COOKIE['user'];
		$connection = mysqli_connect($servername, $username, $password, $database);

		$Details = mysqli_query($connection, "SELECT CourseCode, CourseName, GPA FROM GPA, COURSES WHERE StudentID = $ID AND COURSES.CourseID = GPA.CourseID");

		echo "<table>";

		echo "<tr>
				<th>Course Code</th>
				<th>Course Name</th>
				<th>GPA</th>
			</tr>";

		while ($Data = mysqli_fetch_array($Details)) {
			echo"<tr>
					<td>$Data[CourseCode]</td>
					<td>$Data[CourseName]</td>
					<td>$Data[GPA]</td>
				</tr>";
		}
		
		echo "</table>";
	?>
	</div>

	<div>
	<form action="Add.php" method="POST">
	<div>
		<h2><u>Add New Course</u></h2>

		<h4>Select your course code from the list</h4>

		<?php
			$Courses = mysqli_query($connection, "SELECT CourseID, CourseName, CourseCode FROM COURSES ORDER BY CourseID");

			echo "<select name='course'>";
			while ($Course = mysqli_fetch_array($Courses)) {
				echo "<option value='$Course[CourseID]'>$Course[CourseCode] - $Course[CourseName]</option>";
			}
			echo "</select>"
		?>

	</div>

	<div>
		<select name="gpa">
			<option value="4.0">A+</option>
			<option value="4.0">A</option>
			<option value="3.7">A-</option>
			<option value="3.3">B+</option>
			<option value="3.0">B</option>
			<option value="2.7">B-</option>
			<option value="2.3">C+</option>
			<option value="2.0">C</option>
			<option value="1.7">C-</option>
			<option value="1.3">D+</option>
			<option value="1.0">D</option>
			<option value="0.0">E</option>	
		</select>
	</div>

	<input type="submit" value="Add"/>

	</form>
	</div>

	<div>
	<form action="Remove.php" method="POST">
	<div>
		<h2><u>Remove Course</u></h2>

		<h4>Select your course code from the list</h4>

		<?php
			$Courses = mysqli_query($connection, "SELECT CourseID, CourseName, CourseCode FROM COURSES ORDER BY CourseID");

			echo "<select name='course'>";
			while ($Course = mysqli_fetch_array($Courses)) {
				echo "<option value='$Course[CourseID]'>$Course[CourseCode] - $Course[CourseName]</option>";
			}
			echo "</select>";
			mysqli_close($connection);
		?>

	</div>

	<input type="submit" value="Remove"/>

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