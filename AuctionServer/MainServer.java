import java.io.IOException;
import java.lang.NumberFormatException;

import java.net.ServerSocket;
import java.net.Socket;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;

import java.lang.Double;
import java.util.*;

// Main server handling clients
public class MainServer implements Runnable{

	// Welcoming socket to the clients(fixed)
	public static final int BASE_PORT = 2000;

	// Server socket for main server
	protected static ServerSocket serverSocket = null;

	// Server's database with available items for auction
	public static ItemDB availableItems = null;

	// List to store the data tracked by the main server(bid records)
	private static LinkedList <String> bids;

	// To handle different connection states
	public static final int INITIAL = 0;
	public static final int FIND_ITEM = 1; 
    public static final int ITEM_FOUND = 2;
    public static final int EXIT = 3;

    // Messages to clients depending on their connection states
    public static String WELCOME_MSG = "\nWelcome to Auction Server!!!\n\nEnter command 'quit' at anytime to exit..\n\n"; 
    public static String GET_NAME_MSG = "Please Enter Your Name: "; 
    public static String FIND_ITEM_MSG = "\nPlease Enter Your Symbol: "; 
    public static String ITEM_FOUND_MSG = "\nYour item found in stock.You can bid now!\nCurrent price of the item: "; 
    public static String ITEM_NOT_FOUND_MSG = "\n-1\nThe symbol you entered is not found in stock!\n\nPlease Enter Another Symbol: "; 
    public static String BID_SUCCESS = "\nYour bid was successful!\n"; 
    public static String BID_FAILED = "\nYour bid was not successful!\nPlease Enter A Higher Value:"; 
    public static String BID_ERROR = "\nInvalid Input!!!\nPlease Enter A Number: "; 
    public static String EXIT_MSG = "\nThank you for using Auction Server!!!\nPlease Come Again!!!\n"; 


// Local variables per client(Thread local)

    // Connection socket for the client
	private Socket mySocket;

	// Connection state of the client 
	private int currentState;

	private String clientName = null; // name of the client
	private String item = null;		// bid item of the client
	private Double newPrice = null;		// bid value given by the client
	

	// Constructor for main server
	public MainServer(int socket, ItemDB items) throws IOException {

		serverSocket = new ServerSocket(socket);
		availableItems = items;
		bids = new LinkedList <String>();
		
	}

	// Constructor to give connection of main server to a client
	public MainServer(Socket connectionSocket){

		this.mySocket = connectionSocket;
		this.currentState = INITIAL;
		this.clientName = null;

	}

// Methods to access main servers database
// Synchronized - in order to handle multiple clients
// Static - in order to access global databse of main server
	
	// Returns true if the symbol is found in main servers database
	public static synchronized boolean isValidSymbol(String symbol){

		return availableItems.findKey(symbol);

	}

	// Returns price of a given symbol
	public static synchronized String getPrice(String symbol){
		
		return availableItems.findPrice(symbol);
	}

	// Returns all data of a given symbol
	public static synchronized String [] getData(String symbol){

		return availableItems.findData(symbol);
	}

	// Updating price of a given key
	public static synchronized void updatePrice(String key, String price){

		String [] data = availableItems.findData(key);
		data [1] = price;

		availableItems.updateData(key, data);

	}
// end of synchronized methods

	// Returns security name of a given symbol
	// Not accessed by clients
	public String getSecurity(String key){

		return availableItems.findSecurity(key);

	}

	// Posts the data tracked by the main server on the server's terminal
	//  Adds the tracked data into servers database to keep bid records 
	public void postBID(String string) {

		System.out.println(string);
		bids.add(string);
	}

	// To retriew Server's tracked bid records
	public LinkedList <String> getBids(){

		return this.bids;
	}


	// Main servers functionality
	public static void serverLoop() throws IOException {

		while(true){

			Socket socket = serverSocket.accept();	// trying to connect to the welcoming socket
			Thread client = new Thread(new MainServer(socket)); // Giving connection to a client using a separate thread
			client.start();	// Handling the client
		}
	}

	// Handling the client
	public void run(){

		BufferedReader in = null;
		PrintWriter out = null;

		try{

			in = new BufferedReader(new InputStreamReader(mySocket.getInputStream()));
			out = new PrintWriter(new OutputStreamWriter(mySocket.getOutputStream()));

			String inLine, outLine;

			// Welcome
			out.print(WELCOME_MSG + GET_NAME_MSG);
			out.flush();

			// Getting inputs from the client
			// Client can quit at anytime by entering quit
			for (inLine = in.readLine(); inLine != null && !inLine.equals("quit"); inLine = in.readLine()) {

				// Handling connection states
				switch(this.currentState) { 

					//Get clients name
					case INITIAL:

						if(inLine.equals("")){	// handling empty line

							outLine = GET_NAME_MSG;	//remains in this state until a name is given

						} else {

							this.clientName = inLine;
							outLine = FIND_ITEM_MSG;
							this.currentState = FIND_ITEM;	// assigning next connection state

						}
						
						break;

					// Get valid item symbol 
					case FIND_ITEM: 
					    
					    // waiting for a valid symbol
					    if(isValidSymbol(inLine)) { 

					    	this.item = inLine;	
							this.currentState = ITEM_FOUND; 	// assigning next connection state
							

							// Get old price of the item
							Double oldPrice = Double.parseDouble(getPrice(this.item));

							// Display the price to the client
							outLine = ITEM_FOUND_MSG + oldPrice +"\n\nPlease Enter Your Bid value: "; 

					    } else { 	// Non valid symbol case

							outLine = ITEM_NOT_FOUND_MSG;	// request to enter valid symbol

					    }
					    break;
					
					// Get bid price 
					case ITEM_FOUND: 

						try{	// handling valid inputs for price

							this.newPrice = Double.parseDouble(inLine);

							Double oldPrice = Double.parseDouble(getPrice(this.item));

							// Compare with old price
							if (this.newPrice > oldPrice) {		// valid case

								updatePrice(this.item, inLine);

								// Tracking the changes done by the client 
								postBID(this.clientName + " Bids on: " + this.item +" "+ this.newPrice);

								outLine = BID_SUCCESS + EXIT_MSG;	// Success message to client
								out.print(outLine);

								// Closing the connection with client 
								out.close();
								in.close();
								this.mySocket.close();
							
							} else {	// invalid case

								outLine = BID_FAILED;  
							}

						} catch (NumberFormatException e){	// Invalid input for price case

							outLine = BID_ERROR;

						}
						
					    break; 

					default: 

					    System.out.println("Undefined state\n"); 

					    return; 
					} 

				
				out.print(outLine);
				out.flush();

			}

			// Closing connection if client entered 'quit'
			out.close();
			in.close();
			this.mySocket.close();

		} catch (IOException e) {

			// System.out.println(e);
		}

	}

}