@Reg.No	:E/13/175
@Name	: Kanewala U.C.H.
@Project 1
	
	.text

	.global main

main:

	@ Stack handling
	@ Storing lr
	sub sp,sp,#4
	str lr,[sp,#0]

	@ printf("Enter the key:\n")
	ldr r0,=format1
	BL printf

	@ Storing inputs from commandline
	sub sp,sp,#16

	@ Scanning key1(a)
	@ scanf("%llx", sp)
	ldr r0,=format2
	mov r1,sp
	BL scanf

	@ Load the input key1(a) to registers
	ldr r5,[sp,#0]
	ldr r4,[sp,#4]
	
	@ Scanning key2(b)
	@ scanf("%llx", sp)
	ldr r0,=format2
	mov r1,sp
	BL scanf

	@ Load the input key2(b) to registers
	ldr r7,[sp,#0]
	ldr r6,[sp,#4]
	

	@ printf("Enter the plain text:\n")
	ldr r0,=format3
	BL printf
	
	@ Scanning input pt1(x)
	@ scanf("%llx", sp)
	ldr r0,=format2
	mov r1,sp
	BL scanf

	@ Load the pt1(x) to registers
	ldr r9,[sp,#0]
	ldr r8,[sp,#4]
	
	@ Scanning pt2(y)
	@ scanf("%llx", sp)
	ldr r0,=format2
	mov r1,sp
	BL scanf

	@ Load the pt2(y) to registers
	ldr r11,[sp,#0]
	ldr r10,[sp,#4]
	
	@ R(&x, &y, b)
	@ Loading x and y as arguments to R
	mov r0,r8
	mov r1,r9
	mov r2,r10
	mov r3,r11

	@ Passing key2(b) as an argument to R through stack
	sub sp,sp,#8
	str r6,[sp,#0]
	str r7,[sp,#4]
	BL R 			@ R(&x, &y, b)
	add sp,sp,#8
	
	@ Moving results back to the respective registers of x and y
	mov r8,r0
	mov r9,r1
	mov r10,r2
	mov r11,r3

	@ Setting counter for rounds
	mov r12,#0	@ i =0

	Loop:

		cmp r12,#31
		BGE endLoop

		@ R(&a, &b, i)
		mov r0,#0		@ Setting the first register to zero

		@ Passing i as an argument to R through stack
		sub sp,sp,#8
		str r0,[sp,#0]
		str r12,[sp,#4]

		@ Loading arguments a and b as arguments to R 
		mov r0,r4
		mov r1,r5
		mov r2,r6
		mov r3,r7
		BL R 		@ R(&a, &b, i)
		add sp,sp,#8

		@ Moving results back to the respective registers of a and b
		mov r4,r0
		mov r5,r1
		mov r6,r2
		mov r7,r3

		@ R(&x, &y, b)
		@ Loading arguments x and y as arguments to R 
		mov r0,r8
		mov r1,r9
		mov r2,r10
		mov r3,r11

		@ Passing b as an argument to R through stack
		sub sp,sp,#8
		str r6,[sp,#0]
		str r7,[sp,#4]	
		bl R 			@ R(&x, &y, b) 			
		add sp,sp,#8
		
		@ Moving results back to the respective registers of x and y
		mov r8,r0
		mov r9,r1
		mov r10,r2
		mov r11,r3

		@ Incrementing rounds
		add r12,r12,#1	@ i++
		B Loop

	endLoop:

		@ printf("Cipher text is:\n")
		ldr r0,=format6
		BL printf

		ldr r0,=format5
		mov r1,r9			@passing most significant 32 bits 0f x to r1
		mov r2,r8			@passing least significant 32 bits 0f x to r2
		BL printf

		ldr r0,=format4
		mov r1,r11			@passing most significant 32 bits 0f y to r1
		mov r2,r10			@passing least significant 32 bits 0f y to r2
		BL printf

	@ Restoring stack
	add sp,sp,#16

	@Restoring lr
	ldr lr,[sp,#0]
	add sp,sp,#4
	mov pc,lr
	
ROR:
	
	@ Backing up values in registers used in main
	sub sp,sp,#12
	str r4,[sp,#0]
	str r5,[sp,#4]
	str r6,[sp,#8]


	@ Rotating right 64 bit plain text by r bits 
	@ right shift last 32 bits of x by r bits
	mov r4,r0,lsr r2	@(x >> r)
	rsb r3,r2,#32		@(32 - r)

	@ left shift first 32 bits of x by (32 - r) bits
	mov r5,r1,lsl r3

	@ Combining the two
	orr r4,r4,r5

	@ right shift first 32 bits of x by r bits
	mov r5,r1,lsr r2	@(x >> r)

	@ left shift last 32 bits of x by (32 - r) bits
	mov r6,r0,lsl r3	@(32 - r)

	@ Combining the two
	orr r5,r6,r5

	mov r0,r4
	mov r1,r5

	@ Restoring backed up values
	ldr r4,[sp,#0]
	ldr r5,[sp,#4]
	ldr r6,[sp,#8]
	
	add sp,sp,#12
	mov pc,lr

ROL:

	@ Backing up values in registers used in main
	sub sp,sp,#12
	str r4,[sp,#0]
	str r5,[sp,#4]
	str r6,[sp,#8]

	@ Rotating left 64 bit plain text by r bits 
	@ left shift last 32 bits of y by r bits
	mov r4,r0,lsl r2	@(x << r)
	rsb r3,r2,#32		@(32 - r)

	@ right shift first 32 bits of y by (32 - r) bits
	mov r5,r1,lsr r3

	@ Combining the two
	orr r4,r4,r5

	@ left shift first 32 bits of x by r bits
	mov r5,r1,lsl r2	@(x >> r)

	@ right shift last 32 bits of y by (32 - r) bits
	mov r6,r0,lsr r3

	@ Combining the two
	orr r5,r6,r5

	mov r0,r4
	mov r1,r5

	@ Restoring backed up values
	ldr r4,[sp,#0]
	ldr r5,[sp,#4]
	ldr r6,[sp,#8]
	
	add sp,sp,#12
	mov pc,lr

R:
	
	@ Backing up values in registers used in main
	sub sp,sp,#28
	str lr,[sp,#0]
	str r0,[sp,#4]
	str r1,[sp,#8]
	str r2,[sp,#12]
	str r3,[sp,#16]
	str r4,[sp,#20]
	str r5,[sp,#24]

	@ Loading key2(b) from stack
	ldr r4,[sp,#28]
	ldr r5,[sp,#32]

	@ Storing  key2(b) again in stack
	@str r4,[sp,#28]
	@str r5,[sp,#32]

	mov r2,#8
	mov r3,#0
	BL ROR

	@ Loading pt2(y) from stack
	ldr r2,[sp,#12]
	ldr r3,[sp,#16]

	@ *x += *y
	adds r1,r1,r3
	adc r0,r0,r2

	@ *x ^= *k
	eor r0,r0,r4
	eor r1,r1,r5

	@ Storing resut of pt1(x) to stack
	str r0,[sp,#4]
	str r1,[sp,#8]

	@ *y = ROL(*y,3)
	mov r0,r2
	mov r1,r3
	mov r2,#3
	mov r3,#0
	BL ROL
	mov r2,r0
	mov r3,r1

	@ Loading resut of pt1(x) from stack
	ldr r0,[sp,#4]
	ldr r1,[sp,#8]

	@ *y ^= *x
	eor r2,r2,r0
	eor r3,r3,r1

	@Loading back values stored in the stack
	ldr lr,[sp,#0]
	ldr r4,[sp,#20]
	ldr r5,[sp,#24]

	add sp,sp,#28
	mov pc,lr

	.data

format1: .asciz "Enter the key:\n"
format2: .asciz "%llx"
format3: .asciz "Enter the plain text:\n"
format4: .asciz "%08llx\n"
format5: .asciz "%08llx "
format6: .asciz "Cipher text is:\n"

