<!DOCTYPE html>

<?php
	require 'Credentials.php';	
?>

<html>
<head>
<title>Sign Up</title>
<style type="text/css">
			body{
				margin:0;
				background-image:url("../Images/Pictures/books.jpg");
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

			.forum{
				background-color: white;
				background-image:url("../Images/Pictures/div.jpg");
				position: relative;
				margin: auto;
				top:10px;
    			width: 700px;
				height:1200px ;
				overflow: scroll;
				border-style: solid;
				border-width: 1px;
			}

			.fields{
  			float:right;
  			margin-top: 20px;

			}

			#signInformdiv{
				height: 70px;
				font-size: 50px;
				text-align: center;
				color: white;
				position: relative;
				margin-top: 0px;
				background-color: #cca300;
				border-bottom: solid;
				border-bottom-color: black;

			}

			#register{
				border-radius: 3%;
				background-color:#cca300 ;
				color: black;
				font-size: 20px;
				top:5px;
				left: 0px;
				width: 270px;
				border: none;
				position: relative;
				width: 100px;
				height: 30px;
				

			}

			.Data{
				margin-left: 70px;
				margin-top: 10px;
			}

			#firstName{

				margin-top: 2px;
				width: 300px;	
   				margin-bottom: -1px;
    			border-radius: 5px;
    			border-left: none;
    			border-right: none;
    			border-top: none;
    			border-bottom-color: #000000;
    			background: transparent;

			}

		
			#inputData{
				width: 300px;	
   				margin-bottom: -1px;
    			border-radius: 5px;
    			border-left: none;
    			border-right: none;
    			border-top: none;
    			background: transparent;
    			border-bottom-color: #000000;
			}

			.labels{
				font-size: 20px;
			}

</style>

</head>
	<body>

		<div id ="TopHeader">
			<h1><span font-family: courier>GPA</span> Book</h1>
		</div>

		<div id = "others">
				<a id="aboutGpaBook" href=""><span style="color:white">About GPA Book </span></a>
				<a id="aboutUs" href=""><span style="color:white"> About Us</span></a>
		</div>

		
		<div class ="forum">
			<form class="form-signin" method ="post" action="Register.php">
			<div id ="signInformdiv">	
				<label id="signInform">Sign Up</label><br>
			</div>
			<div id="EnterfirstNamediv" class= "Data">
				<span class ="labels" >Enter your First name : </span><br>
				 <input type="text" name="firstName" id ="firstName" placeholder="Enter your First Name" required>
			</div>	
			<div id="EnterlastNamediv" class= "Data">
				<span class ="labels">Enter your Last name : </span><br>
				 <input type="text" name="lastName" id ="inputData" placeholder="Enter your Last Name" required>
			</div>	
			<div id="initialsdiv" class= "Data">
				<span class ="labels">Enter your Initials : </span><br>
				<input type="text" name="initials" id ="inputData" placeholder="Enter your initials" required>
			</div>

			<div id="gender" class="Data">
				<span class ="labels">Gender : </span><br>
				<input type="radio" name="gender" value="1" checked="checked">Male
				<input type="radio" name="gender" value="0">Female
			</div>

			<div id="universitydiv" class="Data">
				<span class ="labels">Select your University : </span><br>
				<?php

					$connection = mysqli_connect($servername, $username, $password, $database);
					$Universities = mysqli_query($connection, "SELECT UniID, UniName FROM UNIVERSITY ORDER BY UniID");

					echo "<select name='university' id='inputData'>";
					while ($University = mysqli_fetch_array($Universities)) {
						echo "<option value='$University[UniID]'>$University[UniName]</option>";
					}
					echo "</select>";
					mysqli_close($connection);
				?>
			</div>

			<div id="facultydiv" class="Data">
				<span class ="labels">Select your Faculty : </span><br>
				<?php

					$connection = mysqli_connect($servername, $username, $password, $database);
					$Faculties = mysqli_query($connection, "SELECT FacultyID, FacName FROM FACULTY");

					echo "<select name='faculty' id='inputData'>";
					while ($Faculty = mysqli_fetch_array($Faculties)) {
						echo "<option value='$Faculty[FacultyID]'>$Faculty[FacName]</option>";
					}
					echo "</select>";
					mysqli_close($connection);
				?>
			</div>

			<div id="UniIddiv" class= "Data">
				<span class ="labels">Enter your University ID : </span><br>
				<input type="text" name="eno" id ="inputData" placeholder="Ex: E/YY/XXX" required>
			</div>	
		
			<div id="Batchdiv" class= "Data">
				<span class ="labels">Enter your batch : </span><br>
				<input type="text" name="batch" id ="inputData" placeholder="Ex: 2013" required>
			</div>		

			<div id="emaildiv" class= "Data">
				<span class ="labels">Enter your email address : </span><br>
				 <input type="text" name="email" id ="inputData" placeholder="example@mail.com" required>
			</div>

			<div id="passworddiv" class= "Data">
				<span class ="labels">Enter your password : </span><br>
				<input type="password" name="password" id ="inputData" placeholder="need 6 characters" required>
			</div>	

			<div id="passworddiv2" class= "Data">
				<span class ="labels">Re-Enter your password : </span><br>
				<input type="password" name="password2" id ="inputData" placeholder="need 6 characters" required>
			</div>	

			<div id="registerdiv" class= "Data">
				<button id ="register" >Register</button>

			</div>	
			
			</form>
		</div>



</body>
</html>
