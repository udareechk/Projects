/*	CO224 PROJECT 2
	GROUP 07
	GAMLATH P.G.K.R.D.(E/13/110)
	KANEWALA U.C.H. (E/13/175)
*/

// *------------------------------------------------------------- * Processor * ------------------------------------------------------------------*

// Multicycle Processor with 5 stages

/* Processor Module consists of sub modules:
		1. Instruction Fetch
		2. Instruction Decode
		3. Execution
		4. Memory
		5. Write Back
*/

/* Assumptions:
		* All instructions are 16 bit:
				- opcode -> instruction[15:12] 
				- register1 -> instruction[11:8]
				- register2 -> instruction[7:4] 
				- destination register/offset -> instruction[3:0] 
		
		* All registers are 16 bit.
		
		* Memory :
			- 16 bit word size
			- not byte addressed
			
		* ALU:
			- no floating point calculations
		
		* Clock:
			- fixed size clock
			- although different instructions take different number of cycles every instruction takes same time to complete
		

*/

// *----------------------------------------- Instruction Fetch ---------------------------------------------*

// Module for Instruction Fetch of 16 bit instructions
module IF(instruction, nxt_addr, new_addr, jump_addr, jump, clock);

// Declaring input ports as I/O diagram
input [15:0] jump_addr, new_addr;
input jump, clock;

// Declaring output ports as I/O diagram
output [15:0] instruction;
output [15:0] nxt_addr;

// 2D array for Instruction Memory
reg [15:0] icache [127:0];

// Declaring output ports to be registers.
reg [15:0] instruction;
reg [15:0] nxt_addr;

// Loading the program to instruction memory
initial
begin
	
	// The program
	
	// For LW
	icache[0] = 16'h8010;	// LW R0, R1, #0 	--> Load Mem[0] = 10 to R1 ---> R1 = 10
	icache[1] = 16'h80f1;	// LW R0, R15, #1	--> Load Mem[1] = 12 to R15	--> R15 = 12
	
	// For R-type instructions
	icache[2] = 16'h21f3;	// ADD R1, R15, R3	--> R3 = 10 + 12 = 22
    //icache[2] = 16'h61f4;	// SUB R1, R15, R4	--> R4 = 10 - 12 = -2
    //icache[2] = 16'h01f5;	// AND R1, R15, R5	--> R5 = 10 & 12 = 8
    //icache[2] = 16'h1f16;	// OR R1, R15, R6	--> R6 = 10 | 12 = 14
    //icache[2] = 16'h71f7;	// SLT R1, R15, R7	--> R7 = 1
	
	// For SW
	//icache[3] = 16'ha036;	// SW R0, R3, #6	--> Store R3 in Mem[6] --> Mem[6] = 22
	//icache[4] = 16'h80e6;	// LW R0, R14, #6	--> Load Mem[6] = 22 to R14	--> R14 = 22
	
	// For BEQ
	//icache[5] = 16'h8022;	// LW R0, R2, #2	--> Load Mem[2] = 12 to R2	--> R2 = 12
	//icache[6] = 16'he21c; 	// BEQ R2, R1, #-4	--> Branch to #-4 since R2 = R1 = 12 (Branch location: icache[3])
	
	// For Jump
    //icache[6] = 16'hf000;	// JMP R0, #0		-->	Jump to icache[0]
	
	// Initializing instruction memory
	instruction = icache[0];	// Initially giving the first instruction to be icache[0]
	nxt_addr = 0;				// Initially giving the next address to be 0
	
end

// IF Functionality
always @(posedge clock)
begin

	// Fetch instruction after a jump 
	if (jump)
	begin
		instruction = icache[jump_addr];
		nxt_addr = jump_addr + 4'd1;		// Incrementing PC
	end
	
	// Fetch a normal instruction	
	else
	begin
		instruction = icache[new_addr];
		nxt_addr = new_addr + 4'd1;			// Incrementing PC
	end
		
	
end 

endmodule
	
	
// *--------------------------------------------------- Instruction Decode -----------------------------------------------*

/* Assumption: 16 bit opcode -> [15:12] opcode, [11:8] 1st register, [7:4] 2nd register, [3:0] destination register/offset 
*/

/* Instruction Decode Module consists of sub modules:
		1. Control Unit
		2. Register File
		3. Sign Extender
		4. Jump Unit
*/

// --------------------------------------Control Unit-------------------------------------------

// Module for Control Unit
module control_unit(branch, regDest, regWrite, memRead,  memWrite, memToReg, ALUop, ALUsrc, jump, opcode,clock);
 
// Declaring input ports as I/O diagram 
input clock; 
input[3:0] opcode;

// Declaring output ports as I/O diagram
output branch, regDest, memRead, memToReg, memWrite, ALUsrc, regWrite, jump;
output [2:0] ALUop;

// Temporary variables
reg t_branch, t_regDest, t_memRead, t_memToReg, t_memWrite, t_ALUsrc, t_regWrite, t_jump;
reg [2:0] t_ALUop;

// Functionality of Control Unit
always @(posedge clock)
begin
  
	// Giving default values for control lines
 	t_branch = 0; 
	t_regDest = 0; 
 	t_regWrite = 0;
 	t_memRead = 0; 
	t_memWrite = 0;	 
 	t_memToReg = 0; 
 	t_ALUop = 0;  
 	t_ALUsrc = 0;
	t_jump = 0; 
  
  
	case (opcode)
		//AND instruction
    		4'd0 :	begin
			t_regDest = 1;
			t_ALUop = 3'b000;
			t_regWrite = 1;
			end

		//OR instruction
    		4'd1 : 	begin
			t_regDest = 1;
			t_ALUop = 3'b001;
			t_regWrite = 1;
			end

		// ADD instruction
    		4'd2 : 	begin
			t_regDest = 1;
			t_ALUop = 3'b010;
			t_regWrite = 1;
			end

		//SUB instruction
    		4'd6 : 	begin
			t_regDest = 1;
			t_ALUop = 3'b011;
			t_regWrite = 1;
			end

		//SLT instruction
    		4'd7 :	begin
			t_regDest = 1;
			t_ALUop = 3'b111;
			t_regWrite = 1;
			end

		//LW instruction
    		4'd8 : 	begin
			t_ALUsrc = 1;
			t_ALUop = 3'b010;
			t_memRead = 1;
			t_memToReg = 1;
			t_regWrite = 1;
			end

		//SW instruction
    		4'd10 :	begin
			t_ALUsrc = 1;
			t_ALUop = 3'b010;
			t_memWrite = 1;
			end

		//BEQ instruction
    		4'd14 :	begin
			t_ALUop = 3'b011;
			t_branch = 1;
			end
		
		//JMP instruction
   	 	4'd15 : t_jump = 1;

	endcase
end

// Assigning temporary variables to outputs
assign branch = t_branch; 
assign regDest = t_regDest; 
assign regWrite = t_regWrite;
assign memRead = t_memRead; 
assign memWrite = t_memWrite;	 
assign memToReg = t_memToReg; 
assign ALUsrc = t_ALUsrc;
assign ALUop = t_ALUop; 
assign jump = t_jump;
 
endmodule


// --------------------------------------Register File-------------------------------------------

//Module for 16 bit Register File
module reg_file(A, B, C, Addr, Baddr, Cadder, load, clear, clk);

// Declaring input ports as I/O diagram
input [15:0] C;
input [3:0] Cadder, Addr, Baddr;
input load, clear, clk;

// Declaring output ports as I/O diagram
output [15:0] A,B;

//2D array for storage of 16 bit * 16 registers
reg [15:0] data [15:0];

reg [15:0] A,B;
integer i;

// Functionality of Register File
always @(posedge clk or clear)
begin
	if (!clear)
	begin
		for (i = 0; i<16; i=i+1)
			data[i] = 0;	// clear all data
	end
	
	else
	begin
		A = data[Addr];		// read relavent data in given address
		B = data[Baddr]; 	// read relavent data in given address
		
		if (load)
			data[Cadder] = C;	// load data to relavent register
			
	end
end

endmodule


// --------------------------------------Sign Extender-------------------------------------------

// Module for Sign Extender (4 bit ---> 16 bit)
module sign_extender(extn_addr, instruction);

// Declaring input ports as I/O diagram
input [15:0] instruction;
output [15:0] extn_addr;

// Declaring output ports as I/O diagram
reg [15:0] extn_addr;
integer i;

// Functionality of Sign Extender
always @(instruction)
begin

	for (i = 0; i<4; i = i+1)
		extn_addr[i] = instruction[i];
	
	for (i = 4; i<16 ; i = i+1)
		extn_addr[i] = instruction[3];	// replicating the sign bit
	
	//$display("extend addr: %b", extn_addr);
end

endmodule

// --------------------------------------Jump Unit-------------------------------------------

// Module for Jump Unit with 12 bit offset
module jump_unit(jump_addr, instruction, pc);

// Declaring input ports as I/O diagram
input [15:0] instruction;
input [15:0] pc;

// Declaring output ports as I/O diagram
output [15:0] jump_addr;

reg [15:0] jump_addr;

// Functionality of Jump Unit
always @(instruction)
begin
	jump_addr = {pc[15:12], instruction[11:0]};		// jump address calculation
end
	
endmodule

// ---------------------------------ID----------------------------------------------

// Module for Instruction Decode
module ID(jump_addr, branch, regDest, regWrite, memRead,  memWrite, memToReg, ALUop, ALUsrc, jump, read_data1, ALUsrc2, read_data2, extn_addr, instruction, write_data, clock, pc);

// Declaring input ports as I/O diagram
input [15:0] instruction, write_data, pc;
input clock;

// Declaring output ports as I/O diagram
output [15:0] read_data1, jump_addr, ALUsrc2, read_data2, extn_addr;
output [2:0] ALUop;
output branch, regDest, memRead, regWrite, memWrite, memToReg, ALUsrc, jump;

reg [3:0] destination, src1, src2, opcode;
reg [15:0] ALUsrc2;
reg clear;
wire [2:0] ALUop;
//reg [12:0] offset
wire branch, regDest, regWrite, memRead,  memWrite, memToReg, ALUsrc, jump;
wire [15:0] jump_addr;

// Initial conditions for Register File
initial
begin
	clear = 0;		// making all register values zero
	#1 clear = 1'b1;	// clear not activated
end

// Instantiating modules of :
// 1. Control Unit
control_unit control(branch, regDest, regWrite, memRead,  memWrite, memToReg, ALUop, ALUsrc, jump, opcode,clock);
// 2. Register File
reg_file regFile(read_data1, read_data2, write_data, src1, src2, destination, regWrite, clear, clock);
// 3. Sign Extender
sign_extender signextend(extn_addr, instruction);
// 4. Jump Unit
jump_unit myJump(jump_addr, instruction, pc);

// Functionality of Instruction Decode
always @(posedge clock)
begin
	
	opcode = instruction[15:12];
	src1 = instruction[11:8];	// register 1
	src2 = instruction[7:4];	// register 2
	
	// Deciding teh destination register
	if (regDest)
		destination = instruction[3:0];	// For R-type
		
	else
		destination = instruction[7:4];	// For LW, SW, BEQ
	
	// Deciding input for the ALU in EX stage
	if (ALUsrc == 1)
		ALUsrc2 = extn_addr;	// For LW, SW
	else 
		ALUsrc2 = read_data2;	// For R-type, BEQ
		
end

endmodule

// *---------------------------------------------------- Execution ------------------------------------------------------*

/* Execution Module consists of sub module:
		* ALU
*/

// ---------------------------------ALU----------------------------------------------

/* Assumptions:
	* Only unsigned Additions. All the others are signed operations.
	* c_in is ignored in subtract
 */
 
// Module for 16 bit ALU
module alu( z, c_out, lt, eq, gt, overflow, x, y, c_in, control );

// Declaring input ports as I/O diagram
input [15:0] x,y;
input c_in;
input [2:0] control;

// Declaring output ports as I/O diagram
output [15:0] z;
output c_out, lt, eq, gt, overflow;

reg lt, eq, gt, overflow, c_out;
reg [15:0] z;

// ALU function
always @(x, y, c_in, control)
begin
	
	// Setting flags depending on x and y
	lt = ($signed(x) < $signed(y));
	eq = ($signed(x) == $signed(y));
	gt = ($signed(x) > $signed(y));
	overflow = 0;
	
	case(control)
		
		// AND
		 3'b000 : z = x & y;
		 
		// OR
		 3'b001 : z = z | y;
		 
		// ADD
		 3'b010 : 
			begin
				{c_out, z} = x + y + c_in;				
				overflow = (($signed(x) > 0 && $signed(y) > 0 && $signed(z) < 0) || ($signed(x) < 0 && $signed(y) < 0 && $signed(z) > 0));
			end
			
		// SUBTRACT
		 3'b011 : 
			begin
				{c_out, z} = x - y;
				overflow = (x[15] != y[15] &&  x[15] != z[15]); // if x*y<0 then sign(z) == sign(x)
			end
			
		// SLT
		 3'b111 : z = lt;
		 
	endcase
end

endmodule

// ---------------------------------EX----------------------------------------------

// Modue for Execution
module EX(ALUresult, new_addr, ALUop, A, B, extn_addr, nxt_addr, branch, ALUsrc, clock);

// Declaring input ports as I/O diagram
input [15:0] A, B, extn_addr, nxt_addr;
input [2:0] ALUop;
input branch, ALUsrc, clock;

// Declaring output ports as I/O diagram
output [15:0] ALUresult, new_addr;

wire c_out, lt, gt, eq, overflow, c_in;
reg [15:0] Src2, new_addr, shift_addr, branch_addr;

assign c_in = 1'b0;	// initializing c_in

// Instantiating the ALU module
alu myalu(ALUresult, c_out, lt, eq, gt, overflow, A, B ,c_in, ALUop);

// Function of EX
always @(posedge clock)
begin

// Left shifting the sign extended address by 
shift_addr = extn_addr << 1;

// Calculating the branch address
branch_addr = $signed(extn_addr) + nxt_addr;
//$display("ex: %d", $signed(extn_addr));
	
	// BEQ
	if (branch == 1 && eq == 1) 
	begin
		new_addr = branch_addr;
	end
	
	else // No Branch
	begin
		new_addr = nxt_addr;
	end
end

endmodule


// *--------------------------------------------------- Data Memory -------------------------------------------------------*

// Module for Memory with 16 bit word size
module MEM(read_data, ALUresult, write_data, memRead, memWrite, clock);

// Declaring input ports as I/O diagram
input [15:0] write_data, ALUresult;
input memRead, memWrite, clock;

// Declaring output ports as I/O diagram
output [15:0] read_data;

// 2D array for data memory (16 bit wordsize * 127)
reg [15:0] dcache [127:0];
reg [6:0] dcache_addr;

reg [15:0] t_read_data;

// Initializing data memory
initial
begin
	dcache[0] = 16'd10;
	dcache[1] = 16'd12;
	dcache[2] = 16'd10;
	
end

// Functionality of Memory 
always @(posedge clock)
begin
	
	dcache_addr = ALUresult[6:0];
	
	// Memory read
	if (memRead == 1)
		t_read_data = dcache[dcache_addr];
	
	// Memory write
	else if (memWrite == 1)
		dcache[dcache_addr] = write_data;
	
end 

// Getting read data as output 
assign read_data = t_read_data;

endmodule

// *-------------------------------------------------- Write Back -------------------------------------------------------*

// Module for Write Back of a 16 bit data
module WB(write_data, read_data, ALU_result, memToReg, clock);

// Declaring input ports as I/O diagram
input [15:0] read_data;
input [15:0] ALU_result;
input memToReg, clock;

// Declaring output ports as I/O diagram
output [15:0] write_data;

reg [15:0] write_data;

// Functionality of Write Back
always @(posedge clock)
begin

	if(memToReg) // value read from memory
		write_data = read_data;
		
	else // value from ALU 
		write_data = ALU_result;
end

endmodule

// *------------------------------------------------- The Processor ---------------------------------------------------------*

// Module for Processsor
module Processor(instruction, control, read_data1, read_data2, ALUsrc2, ALUresult, read_data, write_data, new_addr, clock);

// Declaring input ports as I/O diagram
input clock;

// Declaring output ports as I/O diagram
output [15:0] instruction, read_data1, read_data2, ALUresult, read_data, write_data, new_addr, ALUsrc2;
output [10:0] control;

wire [15:0] nxt_addr, jump_addr, write_data, ALUsrc2, extn_addr;
wire jump, branch, regDest, regWrite, memRead,  memWrite, memToReg, ALUsrc;
wire [2:0] ALUop; 

assign control = {regDest, jump, branch, memRead, memToReg, ALUop, memWrite, ALUsrc, regWrite};

// variables for clocks
reg clockIF, clockID, clockEX, clockMEM, clockWB;
reg [2:0] clockCount;

// Initializing clocks
initial
begin
	clockIF = 1'b0;
	clockCount = 3'd7;
	
end

// Functionality of Processor
always @(posedge clock)
begin
	clockCount = clockCount + 1;
	if (clockCount == 3'd0)
	begin
		clockIF = ~clockIF;
		$display("clockIF: %b\n", clockIF);
	end
	
	else if (clockCount == 3'd6)
		clockCount = 3'd7;
end

// Instantiating 5 stages of modules:
// 1. IF
IF myIF(instruction, nxt_addr, new_addr, jump_addr, jump, clockIF);
// 2. ID
ID myID(jump_addr, branch, regDest, regWrite, memRead,  memWrite, memToReg, ALUop, ALUsrc, jump, read_data1, ALUsrc2, read_data2, extn_addr, instruction, write_data, clock, nxt_addr);
// 3. EX
EX myEX(ALUresult, new_addr, ALUop, read_data1, ALUsrc2, extn_addr, nxt_addr, branch, ALUsrc, clock);
// 4. MEM
MEM myMEM(read_data, ALUresult, read_data2, memRead, memWrite, clock);
// 5. WB
WB myWB(write_data, read_data, ALUresult, memToReg, clock);

endmodule

// *------------------------------------------------ Simulate Processor --------------------------------------------------------*

//Stimulus module for EX
module stimulusProcessor;

// Declare variables to be connected to inputs
reg CLK;

// Declare output wires
wire [15:0] instruction, read_data1, read_data2, ALUsrc2, ALUresult, read_data, write_data, new_addr;
wire [10:0] control;

// Clock stimulation
initial	
begin
	CLK = 1'b1;
end

always
begin
	#2 CLK = ~CLK;
end

// Instantiating the Processor module
Processor myProcessor(instruction, control, read_data1, read_data2, ALUsrc2, ALUresult, read_data, write_data, new_addr, CLK);


// Stimulate the inputs
initial
begin

	// Monitor changes
	$monitor ("instruction: %h\n Control -> Regdest: %b, Jump: %b, Branch: %b, MemRead: %b, MemToReg: %b, ALUop: %b, MemWrite: %b, ALUsrc: %b, RegWrite: %b\n read_data1: %d\n read_data2: %d\n ALUsrc2: %d\n ALUresult: %d\n read_data: %d\n write_data: %d\n new_addr: %d\n",instruction, control[10], control[9], control[8], control[7], control[6], control[5:3], control[2], control[1], control[0], read_data1, read_data2, ALUsrc2, ALUresult, read_data, $signed(write_data), new_addr);
	// set input lines
	// give nxtAddr
	$display("Clock: %b", CLK);

end

initial
#1500 $stop;
	
endmodule