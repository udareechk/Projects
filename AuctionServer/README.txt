AUCTION SERVER
---------------

This application provides an online auction server to handle multiple clients.

Features:

Server runs on command line interface.
Clients can connect to the server using "nc localhost 2000" via command line interface.
This application provides a GUI to to view the current prices of stock items and their bidding history to the administrator.

Administrators use:

Protocol:

1.	Compile and run AuctionServer.java file use the server as administrator.
2.	Can view current price of an item and history of bids with the item symbol, client's name and value of bids placed by searching 	via the GUI by giving input as the Symbol and clicking the "Search" button
		- Symblo should be given in upper case letters (eg: FB)
3. 	Once the GUI is closed the server will still run but to get the GUI opened again administrator should run the AuctionServer.java
	file again.

Client's use:

Protocol: 

1.	Clients can disconnect from the server at anytime by typing command "quit".
2.	Clients should enter their name first to continue use of the auction server.
3.	Then the clients should enter the symbol of the item they are bidding.
		- Server will indicate whether the symbol is available to the client and if available will display it's current price. 
		- Clients should enter the symbol in uppercase letters.(eg: FB)
		- Client will get chances to input symbols until he/she enters an available item on the auction server.
		- Clients are not allowed to change thei bidding item once they enter a valid symbol(They can quit from the server and connect again if they want to change their bidding item)
4. 	Next the client should enter their bid for the item
		- This should be a valid number(integer or floating point number).
		- The client will get chances to enter their bis untill they enter a valid bidding price.
		- The client will receive messages in any failures throughout the process.
5. 	The client will be automatically disconnected by the auction server once a successful bid is placed for a particular item.
	