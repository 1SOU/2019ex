package stack;

import sun.util.resources.de.CurrencyNames_de_DE;

public class Stack {
    private static class Node{
        private int val;
        private Node next;
        private Node(int x){
            this.val=x;
        }
    }
    private Node top;

    public boolean isEmpty(){

        return top == null;
    }
    public int peek(){//读取栈顶值，不出栈
        // 未考虑 空栈情况，，简化逻辑
        return top.val;
    }
    public int pop(){//出栈
        int value=top.val;
        top=top.next;
        return  value;
    }
    public void push(int value){//入栈
        Node node = new Node(value);
        node.next = top;
        top = node;
    }

    public static void main(String[] args) {
        Stack stack = new Stack();
        System.out.println(stack.isEmpty());

        stack.push(2);
        System.out.println(stack.peek());

    }
}
