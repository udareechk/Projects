//Stimulus module
module stimulusGreyCode;

//Declare variables to be connected as inputs
reg [3:0] IN;
reg S1, S0, ENABLE, CLK;

//Declare variables to be connected as output
wire [3:0] OUT;

//Clock stimulation
initial
begin
	CLK = 1'b0;
end

always
begin
	#1 CLK = ~CLK;
end


//Instantiate the 4 bit Grey_code_counter
grey_code_counter myCounter(OUT, IN, S1, S0, ENABLE, CLK);

//stimulate the inputs
initial 
begin
	//Set input lines
	IN = 4'b1010; ENABLE = 1;
	$display("IN = %b, ENABLE = %b, CLOCK = %b \n", IN, ENABLE, CLK);
	
	
	S1 = 0; S0 = 0;
	#16 $display ("hold: count = 0\nS1 = %b , S0 = %b , OUT = %b \n ", S1, S0, OUT); 
	S1 = 0; S0 = 1;
	#16 $display ("count up: count = 1\nS1 = %b , S0 = %b , OUT = %b \n ", S1, S0, OUT); 
	S1 = 0; S0 = 1;
	#16 $display ("count up: count = 2\nS1 = %b , S0 = %b , OUT = %b \n ", S1, S0, OUT); 
	S1 = 0; S0 = 1;
	#16 $display ("count up: count = 3\nS1 = %b , S0 = %b , OUT = %b \n ", S1, S0, OUT); 
	S1 = 0; S0 = 1;	
	#16 $display ("count up: count = 4\nS1 = %b , S0 = %b , OUT = %b \n ", S1, S0, OUT); 
	S1 = 1; S0 = 0;
	#16 $display ("count down: count = 3\nS1 = %b , S0 = %b , OUT = %b \n ", S1, S0, OUT); 
	S1 = 1; S0 = 0;
	#16 $display ("count down: count = 2\nS1 = %b , S0 = %b , OUT = %b \n ", S1, S0, OUT); 
	S1 = 1; S0 = 1; 
	#16 $display ("Load: count = 10\nS1 = %b , S0 = %b , OUT = %b \n ", S1, S0, OUT); 
	S1 = 0; S0 = 1;
	#16 $display ("count up: count = 11\nS1 = %b , S0 = %b , OUT = %b \n ", S1, S0, OUT); 
	S1 = 0; S0 = 0;
	#16 $display ("hold: count = 11\nS1 = %b , S0 = %b , OUT = %b \n ", S1, S0, OUT);
	S1 = 0; S0 = 1;
	#16 $display ("count up: count = 12\nS1 = %b , S0 = %b , OUT = %b \n ", S1, S0, OUT);

end

initial
#200 $stop;

endmodule

// Module 4-to-1 multiplexer. Port list is taken exactly from
// the I/O diagram.
module mux4_to_1 (out, i0, i1, i2, i3, s1, s0);
	
	// Port declarations from the I/O diagram
	output out;
	input i0, i1, i2, i3;
	input s1, s0;

	reg tempout;
	
	always @(s0,s1,i0,i1,i2,i3)
	begin	
		//$display("mux in: %b, mux control: %b", {i0, i1, i2, i3}, {s1,s0});
		case ({s1,s0})
			2'd0 : tempout = i0;
			2'd1 : tempout = i1;
			2'd2 : tempout = i2;
			2'd3 : tempout = i3;
			default : tempout = 0;	
		endcase
	

	end	
	
	assign out=tempout;
	
endmodule


//Module universal 4 bit shift register
module universal_shift_reg_4(out, in, sl_in, sr_in, s1, s0, enable, clock);
 
output [3:0] out;
input [3:0] in;
input  sl_in, sr_in, s1, s0, enable, clock;

reg [3:0] out;

always @(posedge clock)
begin
	if(enable)
	begin
		case({s1,s0})	
			 2'd0 : out <= out; //HOLD
			 2'd1 : out <= {sr_in, in[3:1]}; //SHIFT RIGHT
			 2'd2 : out <= {in[2:0], sl_in}; //SHIFT LEFT
			 2'd3 : out <= in; //LOAD
		endcase
		//$display("shift in: %b, shift control: %b, shift out: %b", in, {s1,s0}, out);
	end
	else
		out <= out;
end

//assign out = tempout;

endmodule


// Module Grey code converter
// This will convert to grey code any given 4 bit number
module grey_code_converter(out, in, clock);

input [3:0] in;
input clock;
output [3:0] out;

wire [3:0] out, shift1, shift2, shift3;
wire [1:0] c0, c1, c2, c3;

universal_shift_reg_4 r0(shift1, in, 1'b0, 1'b0, 1'b0, 1'b1, 1'b1, clock);	// shift 1
universal_shift_reg_4 r1(shift2, shift1, 1'b0, 1'b0, 1'b0, 1'b1, 1'b1, clock);  // shift 2
universal_shift_reg_4 r2(shift3, shift2, 1'b0, 1'b0, 1'b0, 1'b1, 1'b1, clock);  // shift 3

// select last 2 of shifted numbers
assign c0 = in[1:0];
assign c1 = shift1[1:0];
assign c2 = shift2[1:0];
assign c3 = shift3[1:0];

// select output bits
mux4_to_1 out0(out[0], 1'b0, 1'b1, 1'b1, 1'b0, c0[1], c0[0]);
mux4_to_1 out1(out[1], 1'b0, 1'b1, 1'b1, 1'b0, c1[1], c1[0]);
mux4_to_1 out2(out[2], 1'b0, 1'b1, 1'b1, 1'b0, c2[1], c2[0]);
mux4_to_1 out3(out[3], 1'b0, 1'b1, 1'b1, 1'b0, c3[1], c3[0]);

endmodule


// Module grey code counter
module grey_code_counter(out, in, s1, s0, enable, clock);

input [3:0] in;
input s1, s0, enable, clock;
output [3:0] out;
reg [3:0] count;
reg [2:0] clkCount;

initial
begin

	count = 4'b0;
	clkCount = 2'd3;	// set to 3 because next will be 0;

end

// connect grey code converter
grey_code_converter converter(out, count, clock);

always @(posedge clock)
begin
	// wait for shift registers to complete. They require 3 clock cycles to complete shifting
	clkCount = clkCount + clock;
	
	if(enable && clkCount == 2'b0)
	begin
		case({s1,s0})	
			 2'd0 : count <= count; //HOLD
			 2'd1 : count <= count + 4'b1;  // COUNT UP
			 2'd2 : count <= count - 4'b1; // COUNT DOWN
			 2'd3 : count <= in; //LOAD
		endcase
	end
	else
		count <= count;
end

endmodule