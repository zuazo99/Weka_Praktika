package Hold_Out_For;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class HoldOutFor {
	
	
	private static ArrayList<Double> acc = new ArrayList<Double>();
	
	
	
	public static void main(String[] args) throws Exception {
		
		//HOLD-OUT 5 ALDIZ ERABILITA
		
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1); //Set class atributte.
		
		
		
		for (int i = 0; i < 5; i++) {
			holdOut(data, i);
		}
		
		System.out.println("\nMaximoa: " + Collections.max(acc));
		
		double suma = 0;
		for (Double x : acc) {
			suma += x;
		}
		
		System.out.println("Average: "+ suma/acc.size());
		double average = acc.stream().mapToDouble(a -> a).average().orElse(0.0);
		System.out.println("average java8: " + average);
		
		
		//Variantza
		
		double variance = 0;
		for (Double x : acc){
			variance+= Math.pow(x - average, 2);
		}
		variance/= acc.size();
		
		System.out.println("Std: " + Math.sqrt(variance));
	}
			
	
	
	
	private static void holdOut(Instances data, int i) throws Exception{
			
		Randomize filterRandom = new Randomize();
		filterRandom.setRandomSeed(i); //esto se usa para el for aqui se coloca la i sino hay for se pone 1
		filterRandom.setInputFormat(data); //siempre que modifiques algo le recuerdas al filtro su formato
		Instances RandomData = Filter.useFilter(data, filterRandom);

		RemovePercentage filterRemove = new RemovePercentage();
        filterRemove.setInputFormat(RandomData); //Preparas el filtro.
        filterRemove.setPercentage(30); //Ajustas la cantidad de datos que quieres borrar --> En este caso --> 30% borras y te quedas 70%    
        Instances train = Filter.useFilter(RandomData,filterRemove);
        
        
        filterRemove.setInputFormat(RandomData);
        filterRemove.setPercentage(30);	
        filterRemove.setInvertSelection(true);
        Instances test = Filter.useFilter(RandomData,filterRemove);
        
        
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1 );
        
        
        NaiveBayes model = new NaiveBayes();
        //SMO mirar
        //SMO model1 = new SMO();
        model.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        
     
        eval.evaluateModel(model, test);
        System.out.println(eval.toMatrixString());
        System.out.println(eval.pctCorrect());
        
//        Desviderapena kalkulatu
//        https://stackoverflow.com/questions/21679652/how-to-display-standard-deviation-values-by-using-evaluation-class-of-weka-with
        
        if(data.attribute(data.classIndex()).isNumeric()){
	        AttributeStats as = data.attributeStats(data.classIndex());
	        double std = as.numericStats.stdDev;
	        System.out.println("STD: " + std);
        }
        
        
        //System.out.println("AreaUnderROC: " + eval.areaUnderROC(0));
        acc.add(eval.pctCorrect());
	}
	
			
	private static void fitxategiaSortu(Evaluation eval, String directory){
		
		
		try{
			
			FileWriter file = new FileWriter(directory);
			PrintWriter pw = new PrintWriter(file);
			
			pw.println("Fitxategia sortu:" + directory);
			
			String data = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime());
			pw.println("Exekuzioa data--> "+ data);
			
			pw.println("Nahasmen-Matrizea: " + eval.toMatrixString());
			
			pw.close();
		}catch (Exception e) {
			e.printStackTrace();
			
		}
	}

}
