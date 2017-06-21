import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.*;

import javax.net.ssl.HttpsURLConnection;

public class HttpURLConnectionExample {

	private final String USER_AGENT = "Mozilla/5.0";
	private HashMap<String,String> variables = new HashMap<>();


	public void addVar(String key, String val){
		variables.put(key, val);
	}



	public void sendGet() throws Exception {

		String url = "https://en.wikipedia.org/w/index.php?";

		ArrayList<String> list = new ArrayList<>();

		String tmp;
		for (String x: variables.keySet()){
			tmp = x+"="+variables.get(x);
			list.add(tmp);
		}

		String out = String.join("&", list);

		url += out;

		System.out.println(url);

		URL obj = new URL(url);
		HttpURLConnection con = (HttpURLConnection) obj.openConnection();

		// optional default is GET
		con.setRequestMethod("GET");

		//add request header
		con.setRequestProperty("User-Agent", USER_AGENT);

		int responseCode = con.getResponseCode();
		// System.out.println("\nSending 'GET' request to URL : " + url);
		// System.out.println("Response Code : " + responseCode);

		BufferedReader in = new BufferedReader(
		        new InputStreamReader(con.getInputStream()));
		String inputLine;

		//print result
		StringBuffer response = new StringBuffer();

		while ((inputLine = in.readLine()) != null) {
			response.append(inputLine);
		}
		in.close();
		
		System.out.println(response.toString());
	}


	public static void main(String[] args) throws Exception {

		HttpURLConnectionExample http = new HttpURLConnectionExample();

		http.addVar("search", "rama");
		http.addVar("title", "Special:Search");
		http.addVar("fulltext", "1");
		// http.addVar("searchToken", "6uaz1a4xadtstiar4wudcnjy9");

		// System.out.println("Testing 1 - Send Http GET request");
		http.sendGet();


	}

} 	