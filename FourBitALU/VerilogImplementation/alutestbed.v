// CO221-Digital Design Project phase 1 
// March 2016
// Testbed and the impllementation of a simple ALU 
// This is for those who go beyond the minimal implementation
// and implement the extra part (4 bit muliplier) as well


// Group number : 09
// Names of members : 
// 						U.C.H KANEWALA	(E/13/175)
//						S.L.B. KARUNARATHNE	(E/13/181)
//						S.K.N. KAVINDI (E/13/183) 


//topt level stimulus odule
module testbed;

	reg [3:0] A,B;
	reg load1,load2,run;
	wire [3:0] C; 
	reg signed [3:0] C_twoscomplement;
	reg [1:0] op_select;
	
	//load1 loads A, load2 load B, run gives the output C
	//C is the 4-bit output that comes from the accumulator
	//A and B are the two 4-bit inputs coming from switches
	//load1 is the load signal for operand 1 register
	//load2 is the load signal for operand 2 register
	//run is the load signal for the accumulator register 
	//op_select selects the operator : 00 is xor, 01 is add, 10 is subtract, 11 is multuply 	
	//you don't have to model switches or the 7-segment displays in Verilog
	ALU mu_alu(C,A,B,load1,load2,op_select,run);
	
	//used for representing C as twos complement for the purpose of printing negative numbers
	always @ (C)
		begin	
		C_twoscomplement=C;
	end	
	
	initial
	begin

		//generate files needed to plot the waveform
		//you can plot the waveform generated after running the simulator by using gtkwave
		$dumpfile("wavedata.vcd");
	    $dumpvars(0,testbed);

	
		//A = 5 and B = 2
		A=4'd5; B=4'd2; 
		#5 load1=1'b1;
		#5 load1=1'b0;
		#5 load2=1'b1;
		#5 load2=1'b0;
		#5 op_select=2'b00; 
		#5 run=1'b1;
		#5 run=1'b0;
		$display("%b ^ %b = %b",A,B,C);
		#5 op_select=2'b01; 
		#5 run=1'b1;
		#5 run=1'b0;
		$display("%d + %d = %d",A,B,C);
		#5 op_select=2'b10; 
		#5 run=1'b1;
		#5 run=1'b0; 
		$display("%d - %d = %d",A,B,C_twoscomplement);
		#5 op_select=2'b11; 
		#5 run=1'b1;
		#5 run=1'b0;
		$display("%d * %d = %d\n",A,B,C);

		//A = 2 and B = 5
		#5 A=4'd2; B=4'd5; 
		#5 load1=1'b1;
		#5 load1=1'b0;
		#5 load2=1'b1;
		#5 load2=1'b0;
		#5 op_select=2'b00; 
		#5 run=1'b1;
		#5 run=1'b0;
		$display("%b ^ %b = %b",A,B,C);
		#5 op_select=2'b01; 
		#5 run=1'b1;
		#5 run=1'b0;
		$display("%d + %d = %d",A,B,C);
		#5 op_select=2'b10; 
		#5 run=1'b1;
		#5 run=1'b0;
		$display("%d - %d = %d",A,B,C_twoscomplement);
		#5 op_select=2'b11; 
		#5 run=1'b1;
		#5 run=1'b0;
		$display("%d * %d = %d\n",A,B,C);

		//you can similarly add other testcases.	
		
		
	end
	
endmodule


//your modules should go here

module register(P,Q,load);
	
	input[3:0] P;
	input load;
	output[3:0] Q;
	
	reg[3:0] store;
	assign Q=store;
	
	always @ (posedge load)
	begin
		store=P;
	end
	
endmodule


module ALU(C,A,B,load1,load2,op_select,run);

input [3:0] A;
input [3:0] B;
input [1:0] op_select;

output [3:0] C;

input load1,load2,run;

//define wires
wire [3:0] D;
wire [3:0] X;
wire [3:0] AS;
wire [3:0] M;
wire [3:0] A0;
wire [3:0] B0;
wire [1:0] select;
wire L,w;

//setup the select bus to suit the circuit
and (select[1],op_select[0],op_select[1]);
xor (select[0],op_select[0],op_select[1]);

//setup the input registers
register inp_reg1(A,A0,load1);
register inp_reg2(B,B0,load2);

//setup the control line L which acts as a carry to the adder-subtractor module
not (w,op_select[0]);
and (L,op_select[1],w);

multiplier mult(M,A0,B0);	//setup the multiplier
xor_this x(X,A0,B0);	//setup the xor modules
add_sub as(AS,A0,B0,L);	//setup the adder-subtractor

//declare the input buses to the four 4-to-1 multiplexers
wire [3:0] M0;
wire [3:0] M1;
wire [3:0] M2;
wire [3:0] M3;

//assign values to the multiplexer input buses 
assign M0[0]=X[0];
assign M1[0]=X[1];
assign M2[0]=X[2];
assign M3[0]=X[3];

assign M0[1]=AS[0];
assign M1[1]=AS[1];
assign M2[1]=AS[2];
assign M3[1]=AS[3];

assign M0[2]=M[0];
assign M1[2]=M[1];
assign M2[2]=M[2];
assign M3[2]=M[3];

//setup four 4-to-1 multiplexers
multiplexer_4_1 m0(D[0],M0,select);
multiplexer_4_1 m1(D[1],M1,select);
multiplexer_4_1 m2(D[2],M2,select);
multiplexer_4_1 m3(D[3],M3,select);

register out_reg(D,C,run);	//setup the output register

endmodule


//multiplier
module multiplier(M,A,B);

input [3:0] A;
input [3:0] B;
output [3:0] M;

wire G;	//ground wire
wire [8:0] w;	//internal wiring

assign G=0;	//pull down G to ground

//first AND array
and and0(M[0],A[0],B[0]);
and and1(w[0],A[1],B[0]);
and and2(w[1],A[2],B[0]);
and and3(w[2],A[3],B[0]);

//second AND array
and and4(w[3],A[0],B[1]);
and and5(w[4],A[1],B[1]);
and and6(w[5],A[2],B[1]);

wire [3:0] A0;
wire [3:0] B0;
wire [3:0] S0;
wire CO0;

assign A0[0]=w[0];
assign A0[1]=w[1];
assign A0[2]=w[2];

assign B0[0]=w[3];
assign B0[1]=w[4];
assign B0[2]=w[5];

adder adder0(CO0,S0,A0,B0,G);

assign M[1]=S0[0];

and and7(w[6],A[0],B[2]);
and and8(w[7],A[1],B[2]);

wire [3:0] A1;
wire [3:0] B1;
wire [3:0] S1;
wire CO1;

assign A1[0]=S0[1];
assign A1[1]=S0[2];

assign B1[0]=w[6];
assign B1[1]=w[7];

adder adder1(CO1,S1,A1,B1,G);

assign M[2]=S1[0];

and and9(w[8],A[0],B[3]);

wire [3:0] A2;
wire [3:0] B2;
wire [3:0] S2;
wire CO2;

assign A2[0]=w[8];
assign B2[0]=S1[1];

adder adder2(CO2,S2,A2,B2,G);

assign M[3]=S2[0];

endmodule 


//4-bit adder
module adder(CO,S,A,B,CI);

input [3:0] A;
input [3:0] B;
input CI;

output [3:0] S;
output CO;

wire [2:0] w;	//internal wiring

//setup the line of full adders
full_adder fa0(w[0],S[0],A[0],B[0],CI);
full_adder fa1(w[1],S[1],A[1],B[1],w[0]);
full_adder fa2(w[2],S[2],A[2],B[2],w[1]);
full_adder fa3(CO,S[3],A[3],B[3],w[2]);

endmodule


//setup the adder-subtractor module
module add_sub(AS,A,B,L);

input [3:0] A;
input [3:0] B;
input L;	//L acts as control input which makes this module an adder when L=1 and a subtractor when L=0 

output [3:0] AS;

wire CO;
wire [3:0] D;
wire G;

assign G=0;

xor (D[0],B[0],L);
xor (D[1],B[1],L);
xor (D[2],B[2],L);
xor (D[3],B[3],L);

adder a0(C0,AS,A,D,L);

endmodule 



//xor module
module xor_this(X,A,B);

input [3:0] A;
input [3:0] B;

output [3:0] X;

xor xor0(X[0],A[0],B[0]);
xor xor1(X[1],A[1],B[1]);
xor xor2(X[2],A[2],B[2]);
xor xor3(X[3],A[3],B[3]);

endmodule



//4-to-1 multiplexer module
module multiplexer_4_1(Z,P,Q);

input [3:0] P;
input [1:0] Q;
output Z;

wire [5:0] w;	//internal wiring

not not0(w[0],Q[0]);
not not1(w[1],Q[1]);

and and0(w[2],w[0],w[1],P[0]);
and and1(w[3],w[1],Q[0],P[1]);
and and2(w[4],w[0],Q[1],P[2]);
and and3(w[5],Q[0],Q[1],P[3]);

or (Z,w[2],w[3],w[4],w[5]);

endmodule


//half-adder module
module half_adder(C,S,A,B);

input A,B;
output C,S;

xor (S,A,B);
and (C,A,B);

endmodule



//full-adder module
module full_adder(CO,S,A,B,CI);

input A,B,CI;
output CO,S;

wire [2:0] w;	//internal wiring

//implement using 2 half-adders
half_adder ha0(w[0],w[1],A,B);
half_adder ha1(w[2],S,w[1],CI);
or (CO,w[0],w[2]);

endmodule