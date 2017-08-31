#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched.h>


/* This function is called when the module is loaded. */
int linear_task_lister_init(void)
{
       	printk(KERN_INFO "Loading Linear Task Lister Module\n");

       	struct task_struct *task;

		for_each_process(task){ 
			printk(KERN_INFO "Name: %s\t\t, State: %li\t, PID: [%d]\n", task->comm, task->state, task->pid);
		} 

		return 0;
}

/* This function is called when the module is removed. */
void linear_task_lister_exit(void) {
	printk(KERN_INFO "Removing Linear Task Lister Module\n");
}

/* Macros for registering module entry and exit points. */
module_init( linear_task_lister_init );
module_exit( linear_task_lister_exit );

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Linear Task Lister Module");
MODULE_AUTHOR("UCK");