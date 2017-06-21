import java.io.*;
import java.util.*;
import java.util.Arrays;

// Database for stock items
public class ItemDB{

	private Hashtable <String,String[]> itemMap;

	public ItemDB(String pathToCsv){

		FileReader fRead = null;
		BufferedReader bufRead = null;

		// Read csv file while handling IO exceptions
		try{

			fRead = new FileReader(pathToCsv);
			bufRead = new BufferedReader(fRead);

			itemMap = new Hashtable <String, String[]>();

			//Array to keep one line of csv file
			String [] item;

			//Array to collect value required for the data map
			String [] data;

			for (String line = bufRead.readLine(); line != null; line = bufRead.readLine()) {

				 item = line.split(",");

				 data = new String[item.length -1];

				 // Adding data to be stored as values
				 for (int i = 1; i < item.length; i++ ) {

				 	data[i - 1] = item[i];
				 	
				 }

				// inserting key value pair to Hashtable 
				itemMap.put(item[0], data);
				
				

			}	

		} catch (IOException e){

			System.out.println(e);

		} finally {


		}


	}

	// Returns true if database contains the given key
	public boolean findKey(String key){

		return this.itemMap.containsKey(key);
	}

	// Returns the price of a given key
	public String findPrice(String key){

		String [] value = this.itemMap.get(key);
		Double price = Double.parseDouble(value[1]);

		return Double.toString(price);

	}

	// Returns the security name of a given key
	public String findSecurity(String key){

		String [] value = this.itemMap.get(key);
		return value[0];

	}

	// Returns value of a given key
	public String [] findData(String key){

		return this.itemMap.get(key);

	}

	// Updates key value pair of the map
	public void updateData(String key, String [] data){

		this.itemMap.put(key,data);
	}


} 