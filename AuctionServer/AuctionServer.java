import java.io.IOException;

// Main class for Auction server
public class AuctionServer{

	public static final int BASE_PORT = 2000; // welcoming socket

	// Main method
	public static void main(String[] args) throws IOException {
		
		// Creating databse for auction server
		ItemDB stockDB = new ItemDB("stocks.csv");

		// Creating Main Server to handle Auction Server
		MainServer server = new MainServer(BASE_PORT, stockDB);

		// Create GUI for Auction Server
		DisplayGUI gui = new DisplayGUI(server);

		// handling Auction Server
		server.serverLoop();

	}
}