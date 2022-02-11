package weka_praktikar;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class praktika {

	public static void main(String[] args) throws Exception {
		
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);

		System.out.println("Atributu kop :" + data.numAttributes());
		System.out.println("Instantzia totalak " + data.numInstances());
		System.out.println("Klaseak zenbat balio har ditzake :" + data.numClasses() + " balio hartu ditzake ");
		System.out.println("Klaseak har ditzakeen balio ezberdinak " + data.numDistinctValues(data.classAttribute()));
	
		System.out.println("Azken aurreko atributuak dituen missin value: " 
				+ data.attributeStats(data.numAttributes() - 2).missingCount);
		
		
		
		for (int i = 0; i < data.numAttributes(); i++) {
			System.out.println("Klaseak dituen atributuak: " + data.attribute(i));
		}
		
		
		for (int i = 0; i < data.numAttributes(); i++) {
			System.out.println("Atributu bakoitza hartu diatzakeen baloreak: " + data.attribute(i).numValues());
		}
		
		
		for (int i = 0; i < data.numClasses(); i++) {
			System.out.println("Klasearen balioak , " + i + ". balioa, hurrengoa da " 
		+ data.attribute(data.classIndex()).value(i));

		}
		
		int min = Integer.MAX_VALUE;
		int minClassIndex = 0;
		for (int i = 0; i < data.numClasses(); i++) {
			int x = data.attributeStats(data.classIndex()).nominalCounts[i];
			System.out.println(data.attribute(data.classIndex()).value(i) + "-->" 
			+ x + " instantzia kopurua" );
			
			if(x < min){
				min = x;
				minClassIndex = i;
			}
		}
		
		
		System.out.println("Klase minoritarioa: " + data.attribute(data.classIndex()).value(minClassIndex));
		System.out.println();
		
		
		
		
		// Ebaluazio eskemak
		ebaluazio_ez_zintzoa(data);
		System.out.println();
		System.out.println();
		System.out.println();
		hold_out(data);
		System.out.println();
		System.out.println();
		System.out.println();
		hold_out_Stratified(data);
		System.out.println();
		System.out.println();
		System.out.println();
		k_fold_crossValidation(data, args[1]);
		System.out.println();
		System.out.println();
		System.out.println();
		
	
		
		
		
	}

	private static void ebaluazio_ez_zintzoa(Instances data) throws Exception {

		System.out.println("Ebaluazio ez_zintzoa erabilita :");
		
		NaiveBayes model = new NaiveBayes();
		model.buildClassifier(data);
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(model, data);
		System.out.println("Lortutako datuak :" + eval.toSummaryString());
		System.out.println("Nahasmen matrizea " + eval.toMatrixString());
		System.out.println("Accuracy " + eval.pctCorrect());

	}

	
	private static void hold_out(Instances data) throws Exception {

		System.out.println("Ebaluazio hold out erabilita :");
		
		Randomize randomFilter = new Randomize();
		randomFilter.setInputFormat(data);
		randomFilter.setRandomSeed(1);
		Instances randomData = Filter.useFilter(data, randomFilter);

		RemovePercentage removeFilter = new RemovePercentage();
		removeFilter.setInputFormat(randomData);
		removeFilter.setPercentage(30);

		Instances train = Filter.useFilter(randomData, removeFilter);
		System.out.println("Test instantziak: " + train.numInstances());
		
		

		removeFilter.setInputFormat(randomData);
		removeFilter.setPercentage(30);
		removeFilter.setInvertSelection(true);

		Instances test = Filter.useFilter(randomData, removeFilter);
		System.out.println("Test instantziak: " + test.numInstances());

		train.setClassIndex(train.numAttributes() - 1);
		test.setClassIndex(test.numAttributes() - 1);

		NaiveBayes model = new NaiveBayes();
		model.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(model, test);

		System.out.println("Lortutako datuak :" + eval.toSummaryString());
		System.out.println("Nahasmen matrizea " + eval.toMatrixString());

		System.out.println("Accuracy " + eval.pctCorrect());

	}
	
	private static void hold_out_Stratified(Instances data) throws Exception{
		
		System.out.println("Ebaluazio hold-out-Stratified erabilita :");
		
		//66% train eta 33% test
		StratifiedRemoveFolds filterStratified = new StratifiedRemoveFolds();
		filterStratified.setFold(1); //el fold que me devuelve
		filterStratified.setNumFolds(3); //los fold  que quiero
		filterStratified.setInputFormat(data);
		
		Instances test = Filter.useFilter(data, filterStratified);
		System.out.println("Test instantziak: " + test.numInstances());
		
		
		
		filterStratified.setFold(1);
		filterStratified.setNumFolds(3);
		filterStratified.setInputFormat(data);
		filterStratified.setInvertSelection(true);
		
		Instances train = Filter.useFilter(data, filterStratified);
		System.out.println("Train instantziak: " + train.numInstances());
		
		NaiveBayes model = new NaiveBayes();
		model.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(model, test);
		
		
		System.out.println("Lortutako datuak :" + eval.toSummaryString());
		System.out.println("Nahasmen matrizea " + eval.toMatrixString());

		System.out.println("Accuracy " + eval.pctCorrect());
		
		
		
	}
	

	private static void k_fold_crossValidation(Instances data, String direktorio) throws Exception {
		
		System.out.println("Ebaluazio k-foldCrossValidation erabilita :");

		// Modeloa eraiki Klasifikatzailea
		NaiveBayes model = new NaiveBayes();
		model.buildClassifier(data);

		
		// Sailkatzailea eraiki eta sailkatu
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(model, data, 5, new Random(1));

		System.out.println("Lortutako datuak :" + eval.toSummaryString());
		System.out.println("Nahasmen matrizea " + eval.toMatrixString());
		System.out.println(eval.toClassDetailsString());
		System.out.println("Accuracy: " + eval.pctCorrect());
		System.out.println("Recall: " + eval.recall(0));
		System.out.println("Weighted Recall: " + eval.weightedRecall());
		
		
		fitxategia_idatzi(eval, direktorio);

	}

	private static void fitxategia_idatzi(Evaluation ev, String direktorio)  {
		
		try {
			
			FileWriter file = new FileWriter(direktorio);
			
			PrintWriter pw = new PrintWriter(file);
			
			pw.println("Direktorioa -->" + direktorio);

			pw.println("Nahasmen matrizea -->" + ev.toMatrixString());
			
			pw.close();
		} catch (Exception e) {
			
			e.printStackTrace();
		}
	}

}
