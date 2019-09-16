package heap;

import javax.swing.*;
import java.util.Arrays;

/**
 * max heap?
 */
public class Heap {
    private int capacity = 10; //heap 总大小
    private int size = 0;//heap当前已用空间
    int[] items = new int[capacity];
    // support methods
    private int getParentIndex(int childIndex){return (childIndex-1)/2;}
    private int getLeftChild(int parentIndex){return 2*parentIndex+1;}
    private int getRightChild(int parentIndex){return 2*parentIndex+2;}

    private boolean hasParent(int index){return getParentIndex(index)>=0;}
    private boolean hasLeftChild(int index){return getLeftChild(index)<size;}
    private boolean hasRightChild(int index){return getRightChild(index)<size;}
    private int parent(int index){return items[getParentIndex(index)];}
    private int leftChild(int index){return items[getLeftChild(index)];}
    private int rightChild(int index){return items[getRightChild(index)];}

    private void swap(int index1,int index2){
        int temp = items[index1];
        items[index1]= items[index2];
        items[index2]= temp;
    }
    private void ensureCapacity(){ //size = capacity  ，扩容两倍
        if (size == capacity)
            items = Arrays.copyOf(items, capacity*2);
    }

    public int peek() throws Exception {
        if (size ==0) throw new Exception();
        return items[0];
    }
    public int poll() throws Exception {
        if (size ==0) throw new Exception();
        int item = items[0];
        items[0] =items[size-1]; //删掉根节点的元素，把最后一位放到根节点，在向下交换
        bubbledown();
        return item;
    }
    public void add(int item){
        ensureCapacity();
        items[size] = item;
        size++;
        bubbleup();
    }

    public void bubbleup(){
        int index = size-1;//最后一位，向上交换
        while (hasParent(index) && parent(index)>items[index]) {
            swap(getParentIndex(index), index);
            index= getParentIndex(index);
        }
    }
    public void bubbledown(){
        int index=0;
        while (hasLeftChild(index)){
            int smallIndex=getLeftChild(index);
            if (hasRightChild(index) && rightChild(index) < leftChild(index))
                smallIndex = getRightChild(index);
            if (items[index] < items[smallIndex]) {
                break;
                //不用再交换，根节点已经是最小..退出循环
            }
            else
                swap(index, smallIndex);
            index = smallIndex;
        }
    }

    public static void main(String[] args) throws Exception {
        Heap heap =new Heap();

        heap.add(40);
        heap.add(30);
        heap.add(5);
        heap.add(60);
        heap.add(6);

        System.out.println(heap.peek());
        heap.poll();
        System.out.println(heap.peek());

    }
}
