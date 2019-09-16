package hashmap;

import java.util.HashMap;
import java.util.Map;

/**
 * No.1  Two Sum
 *  * Given an array of integers, return indices of the two numbers such that they add up to a specific target.
 *  * You may assume that each input would have exactly one solution, and you may not use the same element twice.
 *  * Example:
 *  * Given nums = [2, 7, 11, 15], target = 9,
 *  * Because nums[0] + nums[1] = 2 + 7 = 9,
 *  * return [0, 1].
 *  *
 *  * solution
 *  构造哈希表  O(n)
 *  * key   value
 *  * 2        0
 *  * 7        1
 *  * 11       2
 *  * 15       3
 *  * 哈希map中，hashMap.containsKey()是 O(1)，按需求确定谁是 key
 */
public class TwoSum {
    private static int[] twosum(int[] numbers, int target){
        //define return
        int[] indexArray = new int[2];
        //handle corner cases
        if (numbers == null || numbers.length == 0)
            return null;

        //value--> index map
        Map<Integer,Integer> hashMap = new HashMap<>();
        for (int i =0; i<numbers.length;i++){
            hashMap.put(numbers[i],i);
        }
        for (int i=1;i<numbers.length;i++){
            int compl = target-numbers[i];
            if (hashMap.containsKey(compl)){
                indexArray[0] = i;
                indexArray[1] = hashMap.get(compl); //由key 获取value
            }
            if (indexArray[0] == indexArray[1])
                continue; //题目已确定数字不会重复。规定一个数字只能用一次，
                          // 但是防止某个数字 key +value = target 的情况
            return indexArray;
        }
        return null; // 遍历完 ，没有符合
    }
}
