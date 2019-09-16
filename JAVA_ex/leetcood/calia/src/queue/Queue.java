package queue;

import sun.awt.windows.WPrinterJob;

public class Queue {
    private static class Node{
        private int val;
        private Node next;
        private Node(int x){
            this.val=x;
        }
    }
    private Node head;
    private Node tail;

    public boolean isEmpty(){
        return head == null;
    }
    public int peek(){//返回队头值
        //为考虑 队空的情况
        return head.val;
    }
    public void add(int value){//入队 ,,在队尾
        Node node = new Node(value);
        if (tail != null) {
            tail.next = node;
        }
        tail = node;
        if (head == null)
            head =node;
    }
    public int remove(){//出队，在队头，并返回值
        //未考虑队空的情况
        int value = head.val;
        head=head.next;//当只有一个元素，出队之后。队空，head 指向null
        //tail也要指向null
        if (head== null)
            tail = null;
        return value;
    }

    public static void main(String[] args) {
        Queue queue = new Queue();
        System.out.println(queue.isEmpty());
        queue.add(4);
        queue.add(55);
        System.out.println(queue.peek());
        System.out.println(queue);
    }
}
