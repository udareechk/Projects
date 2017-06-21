import java.awt.*;
import javax.swing.Timer;

import java.awt.BorderLayout;
import javax.swing.*;

import java.io.IOException;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.awt.event.*;

// To handle GUI
public class DisplayGUI extends JPanel implements ActionListener {

	
	private MainServer server;	// server for the GUI

	private JLabel [][] labels;	// to keep required default items
	private String [] symbols;	// to keep required symbols of default items

	private JTextField input;	// to get search input
	private JButton search;		// button to search

	// Constructor
	public DisplayGUI(MainServer svr){
		super(new BorderLayout());

		this.server = svr;	

		// Display elements for GUI
		JFrame frame = new JFrame("Auction Server");

		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

		JPanel gui = new JPanel(new BorderLayout());	// GUI panel

		// panel to display default items
		JPanel mainPanel = new JPanel(new GridLayout(8,3));	

		symbols = new String []{"FB", "VRTU", "MSFT", "GOOGL", "YHOO", "XLNX", "TSLA", "TXN"};	//Default symbols


		JPanel [][] panels = new JPanel [8][3];
		labels = new JLabel [8][3];

		for (int i = 0; i < 8 ; i++) {

			for (int j = 0; j<3 ; j++ ) {

				panels[i][j] = new JPanel();
				mainPanel.add(panels[i][j]);
				
			}
		}

		// Adding default items and their data
		for (int i = 0; i < 8 ; i++) {

			// symbol
			labels[i][0] = new JLabel(symbols[i], SwingConstants.CENTER);
			panels[i][0].add(labels[i][0], BorderLayout.CENTER);
			panels[i][0].setPreferredSize(new Dimension(100, 50));

			// security name
			labels[i][1] = new JLabel(server.getSecurity(symbols[i]), SwingConstants.CENTER);
			panels[i][1].setPreferredSize(new Dimension(400, 50));
			panels[i][1].add(labels[i][1], BorderLayout.CENTER);

			// price
			labels[i][2] = new JLabel(server.getPrice(symbols[i]), SwingConstants.CENTER);
			panels[i][2].add(labels[i][2], BorderLayout.CENTER);
			panels[i][2].setPreferredSize(new Dimension(100, 50));

		}

		gui.add(mainPanel, BorderLayout.NORTH);	// add to gui panel
		
		// Search panel
		JPanel searchPanel = new JPanel();

		JLabel symbol = new JLabel("Enter Symbol :", SwingConstants.CENTER);

		input = new JTextField(10);

	 	search = new JButton("Search");
		search.addActionListener(new SearchController(server, search, input));	// add a different action listener

		searchPanel.add(symbol, BorderLayout.EAST);
		searchPanel.add(input, BorderLayout.CENTER);
		searchPanel.add(search, BorderLayout.WEST);

		
		gui.add(searchPanel, BorderLayout.SOUTH);	// add to gui panel


		frame.setContentPane(gui);	// add gui panel to main frame

		frame.pack();
		frame.setVisible(true);
		
		// timer to update GUI every 500 ms
		Timer timer = new Timer(500, this);	// adding task	
		timer.start();

	}


	// updating prices of default items according to the action of timer
	public void actionPerformed(ActionEvent e) { 

		for (int i = 0; i < 8 ; i++) {

			String newPrice = server.getPrice(symbols[i]);
			labels[i][2].setText(newPrice);

		}

		
	}

}