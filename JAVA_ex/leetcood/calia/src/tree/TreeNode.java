package tree;

public class TreeNode {
    TreeNode left;
    TreeNode right;
    int val;

    public TreeNode(int x){ //构造器
        this.val = x;
    }

    public void insert(int value){
        if (value <= val){ //小的放左边，大的放右边
            if (left == null)
                left = new TreeNode(value);
            else
                left.insert(value);
        }
        else
            if (right==null)
                right = new TreeNode(value);
            else
                right.insert(value);
    }
    public boolean contains(int value){
        if (value == val)
            return true;
        else if (value < val){
            if (left == null)
                return false;
            else return left.contains(value);}
        else {
            if (right == null)
                return false;
            else return right.contains(value);
        }
    }
    public void printInorder(){//中序遍历打印
        if (left != null)
            left.printInorder();
        System.out.println(val);
        if (right != null)
            right.printInorder();
    }

    public static void main(String[] args) {
        TreeNode t1 = new TreeNode(1);
        t1.insert(55);
        t1.insert(0);
        System.out.println(t1.val);//输出根节点的值
        System.out.println(t1.contains(55));
        t1.printInorder();
    }
}
