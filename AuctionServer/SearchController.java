import java.awt.BorderLayout;
import javax.swing.*;
import java.awt.*;

import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;

import java.util.*;
import java.lang.Exception;

//  Controller to perform search functionality of the GUI
public class SearchController extends JPanel implements ActionListener{

	private MainServer server;	// server for the Search Controller
	private JButton search;
	private JTextField input;
	private JTextArea txtArea;

	// messages for to be displayed
	private final String CURRENT_PRICE_MSG = "\nCurrent price of the item ";
	private final String HISTORY_MSG = "\nHistory of bids: \n\n";
	private final String NO_BIDS_MSG = "No previous bids found for this item.\n";
	private final String NO_ITEM_MSG = "\nThe item you searched is not found in stock!\nPlease try another Symbol..\n";

	// Constructor
	public SearchController(MainServer svr, JButton button, JTextField text){

		this.server = svr;
		this.search = button;
		this.input = text;

	}

	// Handling search 
	public void actionPerformed(ActionEvent e) { 

		if ((JButton)e.getSource() == search) {	// check for click on search button

			String symbol = input.getText();	// get text in text field

			DisplayItem(symbol);	
		}

	}


	// method to find item and display it's price and bidding history 
	public void DisplayItem(String symbol){

		// Craeting new window
		JFrame frame2 = new JFrame("Item Results");

		frame2.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

		JPanel resultPanel = new JPanel();

		txtArea = new JTextArea(30,40);
		JScrollPane scrollPane = new JScrollPane(txtArea); // To handle text if the given text area is not enough
		txtArea.setEditable(false);	// making not editable to the user

		
		// Check for availability of the symbol in servers database
		if (server.isValidSymbol(symbol)) {

			String price = server.getPrice(symbol);
			txtArea.append(CURRENT_PRICE_MSG + symbol+ " : "+ price+ "\n");

			txtArea.append(HISTORY_MSG);

			// Getting tracked bid records from server
			LinkedList <String> bids = server.getBids();

			int count = 0;	//Counter for records

			if (bids != null) {	//Non empty record history

				for (String s : bids) {

					if (s.contains(symbol)) {

						txtArea.append( s + "\n");	// view records found
						count++;

					} 
				}

				if (count == 0) {	//No records found
					
					txtArea.append(NO_BIDS_MSG);
				}
				
			} else {	// No record history
				
				txtArea.append(NO_BIDS_MSG);
			}


		} else	{ // symbol not found in server's database

			txtArea.append(NO_ITEM_MSG);
		}

		
		resultPanel.add(txtArea);

		frame2.setContentPane(resultPanel);	//adding content

		frame2.pack();
		frame2.setVisible(true);

	}

}
