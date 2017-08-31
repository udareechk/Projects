#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched.h>

void depth_first_search(struct task_struct *task);

/* This function is called when the module is loaded. */
int dfs_task_lister_init(void)
{
    
    printk(KERN_INFO "Loading DFS Task Lister Module\n");

    depth_first_search(&init_task);   	


	return 0;
}

/* This function is called when the module is removed. */
void dfs_task_lister_exit(void) {
	printk(KERN_INFO "Removing Linear Task Lister Module\n");
}

/* This function recursively calls itself  */
void depth_first_search(struct task_struct *parent){

	printk(KERN_INFO "Name: %s\t\t, State: %li\t, PID: [%d]\n", parent->comm, parent->state, parent->pid);
	
	struct task_struct *child;
	struct list_head *list;

	list_for_each(list, &parent -> children){ 
		// Get the next child in the list
		child = list_entry(list, struct task_struct, sibling);
		depth_first_search(child);
	} 
	
}

/* Macros for registering module entry and exit points. */
module_init( dfs_task_lister_init );
module_exit( dfs_task_lister_exit );

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("DFS Task Lister Module");
MODULE_AUTHOR("UCK");