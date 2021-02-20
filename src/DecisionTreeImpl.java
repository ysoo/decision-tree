import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Fill in the implementation details of the class DecisionTree using this file. Any methods or
 * secondary classes that you want are fine but we will only interact with those methods in the
 * DecisionTree framework.
 * 
 * Name : Yi Xian Soo
 * CS 540 Homework 3 Problem 3
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl{
	private DecTreeNode root;
	//ordered list of attributes
	private ArrayList<String> mTrainAttributes; 
	private ArrayList<ArrayList<Double>> mTrainDataSet;
	//Min number of instances per leaf.
	private int minLeafNumber = 10;
	private double threshold;
	private int bestAttribute;

	/**
	 * Answers static questions about decision trees.
	 */
	DecisionTreeImpl() {
		// no code necessary this is void purposefully
	}

	/**
	 * Initializes decision tree. Calls buildTree() to initialize the root member.
	 * Build a decision tree given a training set then prune it using a tuning set.
	 * 
	 * @param train: the training set
	 * @param tune: the tuning set
	 */
	DecisionTreeImpl(ArrayList<ArrayList<Double>> trainDataSet, ArrayList<String> trainAttributeNames, int minLeafNumber) {
		this.mTrainAttributes = trainAttributeNames;
		this.mTrainDataSet = trainDataSet;
		this.minLeafNumber = minLeafNumber;
		this.root = buildTree(this.mTrainDataSet);
	}

	/**
	 * Recursive function that is used to build decision tree. 
	 * @param dataSet
	 * @return DecTreeNode
	 */
	private DecTreeNode buildTree(ArrayList<ArrayList<Double>> dataSet){
		// Base cases
		
		DecTreeNode root = new DecTreeNode(pluralityValue(dataSet), mTrainAttributes.get(bestAttribute), threshold);
		
		if(dataSet.isEmpty() || dataSet.size() <= minLeafNumber || sameClass(dataSet))  {
			root.right = null;
			root.left = null;
			return root;
		} else {
			// Call rootInfoGain to update index threshold and attributes
			rootInfoGain(dataSet, mTrainAttributes, minLeafNumber, false);
			root.attribute = mTrainAttributes.get(bestAttribute);
			root.threshold = threshold;
			
			//System.out.println("Best Attribute: " + bestAttribute + "  Threshold: " + threshold);

			ArrayList<DataBinder> datasort = new ArrayList<DataBinder>();
			for(int i = 0; i < dataSet.size(); i ++) {
				DataBinder row = new DataBinder(bestAttribute, dataSet.get(i));
				datasort.add(row);
			}
			DBComparator comparator = new DBComparator();
			Collections.sort(datasort, comparator);

			ArrayList<DataBinder> leftdata = new ArrayList<DataBinder>();
			ArrayList<DataBinder> rightdata = new ArrayList<DataBinder>();
			for(int i = 0 ; i < dataSet.size(); i ++) {
				DataBinder temp = datasort.get(i);
				double value = temp.getArgItem();
				if(value <= threshold) {
					leftdata.add(temp);
				} else if (value > threshold) {
					rightdata.add(temp);
				}
			}

			// TODO: Get the attributes and split data into two sets 
			ArrayList<ArrayList<Double>> leftSet = new ArrayList<ArrayList<Double>>();
			ArrayList<ArrayList<Double>> rightSet = new ArrayList<ArrayList<Double>>();

			for(int i = 0; i < leftdata.size(); i ++){
				leftSet.add(leftdata.get(i).getData());
			
			}
	
			for(int i = 0; i < rightdata.size(); i ++){
				rightSet.add(rightdata.get(i).getData());
			
			}
			// Call buildTree on left than right tree

			root.left = buildTree(leftSet);
			root.right = buildTree(rightSet);

			// Create DecTreeNode
			return root;
		}
	}

	private int pluralityValue(ArrayList<ArrayList<Double>> dataSet) {
		int zero = 0;
		int one = 0;
		for(int i = 0; i < dataSet.size(); i ++) {
			if(dataSet.get(i).get(mTrainAttributes.size()) == 0) {
				zero++;
			} else if (dataSet.get(i).get(mTrainAttributes.size()) == 1) {
				one++;
			}
		}
		if(zero > one) return 0;
		else return 1;

	}
	
	private boolean sameClass(ArrayList<ArrayList<Double>> dataSet) {
		boolean temp = true;
		double holder = dataSet.get(0).get(mTrainAttributes.size());
		for(int i = 0; i < dataSet.size(); i ++) {
			double curr = dataSet.get(i).get(mTrainAttributes.size());
			if(curr != holder) {
				temp = false;
			}
		}
		return temp;
	}

	/**
	 * Predicts the label (0 or 1) for a provided instance, using the already constructed
	 * decision tree
	 * 
	 * @param instance 
	 * @return label
	 */
	public int classify(List<Double> instance, DecTreeNode root) {
		
		if(root.isLeaf()){
			return root.classLabel;
		}
		
		// Find index of attribute of root 
		int rootAtt = 0;
		for(int i = 0; i < mTrainAttributes.size(); i ++) {
			if(root.attribute.equals(mTrainAttributes.get(i))) {
				rootAtt = i;
			}
		}
		
		if(instance.get(rootAtt) <= root.threshold){
			return classify(instance, root.left);
		}else{
			return classify(instance, root.right);
		}
	}
	
	public DecTreeNode getRoot(){
		return this.root;
	}
	
	public double printAccuracy(int numEqual, int numTotal){
		double accuracy = numEqual/(double)numTotal;
		System.out.println(accuracy);
		return accuracy;
	}

	/**
	 *  Prints the best information gain that could be achieved by splitting on a 
	 *  an attribute, for all the attributes at the root (one in each line), using 
	 *  the training set
	 *  
	 * @param dataSet	training set
	 * @param trainAttributeNames	training attributes
	 * @param minLeafNumber		minimum leaf number
	 */
	public void rootInfoGain(ArrayList<ArrayList<Double>> dataSet, ArrayList<String> trainAttributeNames, int minLeafNumber, boolean print) {
		this.mTrainAttributes = trainAttributeNames;
		this.mTrainDataSet = dataSet;
		this.minLeafNumber = minLeafNumber;

		// 0  - Information Gain ; 1 - Threshold; 2 - Entropy ; 3 - Index in dataSet // For debugging purposes
		double[][] infoGain = new double[mTrainAttributes.size()][4];
		// ITERATE OVER NUMBER OF COLUMNS(size of trainattributes)
		for(int j = 0; j < mTrainAttributes.size(); j ++) { //current attribute
			ArrayList<DataBinder> datasort = new ArrayList<DataBinder>();
			// TODO: Sort instances by value
		
			for(int i = 0; i < dataSet.size(); i ++) {
				DataBinder row = new DataBinder(j, dataSet.get(i));
				datasort.add(row);
			}
			DBComparator comparator = new DBComparator();
			Collections.sort(datasort, comparator);

			//Checking methods
			for(int i = 0; i < datasort.size(); i ++) {
				DataBinder temp = datasort.get(i);
				//System.out.println(temp.getArgItem() + " " +  temp.getmClass());
			}


			// TODO: Find consecutive instances with different class labels
			double holder = datasort.get(0).getmClass();
			int zeros = 0;
			if (holder == 0) zeros++; 
			ArrayList<Double> thresholds = new ArrayList<Double>();
			for(int i = 1 ; i < datasort.size(); i ++) {
				DataBinder temp = datasort.get(i);
				double curr = temp.getmClass();

				if(curr == 0)
					zeros++;

				// TODO: Calculate candidate threshold put into a list 
				if(curr != holder) {
					double threshold = threshold(temp.getArgItem(), datasort.get(i - 1).getArgItem());
					thresholds.add(threshold);
				}	
				holder = curr;
			}

			// TODO: Calculate H(class)

			double Hclass = entropy((double)zeros/(double)datasort.size(), ((double)datasort.size() - (double)zeros)/(double)datasort.size());
				System.out.println("hclass " + Hclass);
			ArrayList<Double> entropy = new ArrayList<Double>();


			// TODO: For each threshold, split data into 2 lists : L1 <=threshold, L2 > threshold
			Collections.sort(thresholds);
			
			for(int i = 0; i < thresholds.size(); i ++) {
				double thisone = thresholds.get(i);
				ArrayList<DataBinder> list1 = new ArrayList<DataBinder>();
				ArrayList<DataBinder> list2 = new ArrayList<DataBinder>();
				int zero1 = 0;
				int zero2 = 0;
				
				for(int k = 0; k < datasort.size(); k++) {
					DataBinder temp = datasort.get(k);
					if(temp.getArgItem() <= thisone) {
						list1.add(temp);
						if(temp.getmClass() == 0) zero1++;
					} else if (temp.getArgItem() > thisone) {
						list2.add(temp);
						if(temp.getmClass() == 0) zero2++;
					}
				}
				
				// TODO: Calculate entropy for the split
				// Repeat for ALL splits and stick it in a list
				double P1 = (double)list1.size()/ (double)datasort.size();
				double P2 = (double)list2.size()/(double)datasort.size();
				double H1 = entropy((double)zero1/(double)list1.size(), ((double)list1.size() - (double)zero1)/(double)list1.size());
				double H2 = entropy((double)zero2/(double)list2.size(), ((double)list2.size() - (double)zero2)/(double)list2.size());
				double H = P1*H1 + P2*H2;
				System.out.print("H: " + H);
				entropy.add(H);

				System.out.println(" threshold: " + thisone);
			}

			// TODO: Find smallest entropy and I(class) - H (Class | A1)
			double minentropy = Collections.min(entropy);
			double gain = Hclass - minentropy;
			System.out.println(Collections.min(entropy));

			// Get max entropy threshold
			for(int i = 0; i < entropy.size(); i ++) {
				if(minentropy == entropy.get(i)){
					infoGain[j][1] = (double)thresholds.get(i);
					infoGain[j][2] = entropy.get(i);
				}
			}
			infoGain[j][0] = gain;
			System.out.println(gain);
		}
		//TODO: modify this example print statement to work with your code to output attribute names and info gain. Note the %.6f output format.

		if(print) {
			for(int i = 0; i<infoGain.length; i++){
				System.out.println(this.mTrainAttributes.get(i) + " " + String.format("%.6f", infoGain[i][0]));
			}
		}
		double gains = infoGain[0][0];
		for(int k = 0; k < infoGain.length; k ++) {
			if(infoGain[k][0] >= gains) {
				gains = infoGain[k][0];
				bestAttribute = k;
				threshold = infoGain[k][1];
			}
		}
	}

	private double threshold(double num1, double num2) {
		return(num1 + num2)/2;
	}
	private double entropy(double num1, double num2) {
		try {
			double entropy = -(num1*log(num1) )- (num2*log(num2));
			return entropy;
		} catch (ArithmeticException ae) {
			return 0;
		}
	}
	private double log(double x) {
		double logx = Math.log(x);
		double log2 = Math.log(2);
		double value = logx/log2;
		if (value == Double.NEGATIVE_INFINITY || value == Double.POSITIVE_INFINITY)
			return 0;
		return value;
	}

	/**
	 * Print the decision tree in the specified format
	 */
	public void print() {
		printTreeNode("", this.root);
	}

	/**
	 * Recursively prints the tree structure, left subtree first, then right subtree.
	 */
	public void printTreeNode(String prefixStr, DecTreeNode node) {
		String printStr = prefixStr + node.attribute;

		System.out.print(printStr + " <= " + String.format("%.6f", node.threshold));
		if(node.left.isLeaf()){
			System.out.println(": " + String.valueOf(node.left.classLabel));
		}else{
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.left);
		}
		System.out.print(printStr + " > " + String.format("%.6f", node.threshold));
		if(node.right.isLeaf()){
			System.out.println(": " + String.valueOf(node.right.classLabel));
		}else{
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.right);
		}


	}

	/**
	 * Private class to facilitate instance sorting by argument position since java doesn't like passing variables to comparators through
	 * nested variable scopes.
	 * */
	private class DataBinder{

		public ArrayList<Double> mData;
		public int i;
		public int size = mTrainAttributes.size();
		public DataBinder(int i, ArrayList<Double> mData){
			this.mData = mData;
			this.i = i;
		}
		public double getArgItem(){
			return mData.get(i);
		}
		public double getmClass() {
			return mData.get(size);
		}
		public ArrayList<Double> getData(){
			return mData;
		}
	}

	private class DBComparator implements Comparator<DataBinder> {

		@Override
		public int compare(DataBinder o1, DataBinder o2) {
			if(o1.getArgItem() < o2.getArgItem()) 
				return -1;
			else if(o1.getArgItem() ==  o2.getArgItem()) {
				if(o1.getmClass() < o2.getmClass()) {
					return -1;
				} else if(o1.getmClass() > o2.getmClass()) {
					return 1;
				}
				else return 0;
			}
			else return 1;
		}
	}
}

