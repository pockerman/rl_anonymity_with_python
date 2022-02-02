/**
 *  Investigate various output quality measures supplied by ARX
 *
 */
package examples.example_3


import base.DefaultConfiguration
import org.deidentifier.arx.AttributeType.{Hierarchy, listMicroAggregationFunctions}
import org.deidentifier.arx.aggregates.AggregateFunction
import org.deidentifier.arx.aggregates.AggregateFunction.AggregateFunctionBuilder
//import org.deidentifier.arx.aggregates.AggregateFunction.AggregateFunctionBuilder.*
import org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased
import org.deidentifier.arx.criteria.KAnonymity
import org.deidentifier.arx.{ARXAnonymizer, ARXConfiguration, AttributeType, Data, DataType}

import java.lang
import collection.JavaConverters.*
import collection.convert.ImplicitConversions.*
import java.io.File
import java.nio.charset.{Charset, StandardCharsets}
import postprocessor.ResultPrinter.{printHandle, printHandleTop, printResult}


object MeasureDataQuality extends App{

  def buildSalaryHierarchy: HierarchyBuilderIntervalBased[lang.Double] = {

    val salaryHierarchy: HierarchyBuilderIntervalBased[lang.Double] = HierarchyBuilderIntervalBased.create(DataType.DECIMAL)
    val aggregateFunctionBuilder = AggregateFunction.forType(DataType.DECIMAL)

    salaryHierarchy.addInterval(lang.Double(0.0), lang.Double(0.2222222222222222), aggregateFunctionBuilder.createArithmeticMeanOfBoundsFunction())
    salaryHierarchy.addInterval(lang.Double(0.2222222222222222), lang.Double(0.4444444444444444), aggregateFunctionBuilder.createArithmeticMeanOfBoundsFunction())
    salaryHierarchy.addInterval(lang.Double(0.4444444444444444), lang.Double(0.6666666666666666), aggregateFunctionBuilder.createArithmeticMeanOfBoundsFunction())
    salaryHierarchy.addInterval(lang.Double(0.6666666666666666), lang.Double(0.8888888888888888), aggregateFunctionBuilder.createArithmeticMeanOfBoundsFunction())
    salaryHierarchy.addInterval(lang.Double(0.8888888888888888), lang.Double(1.1111111111111112), aggregateFunctionBuilder.createArithmeticMeanOfBoundsFunction())
    salaryHierarchy.addInterval(lang.Double(1.1111111111111112), lang.Double(1.3333333333333333), aggregateFunctionBuilder.createArithmeticMeanOfBoundsFunction())
    salaryHierarchy.addInterval(lang.Double(1.3333333333333333), lang.Double(1.5555555555555554), aggregateFunctionBuilder.createArithmeticMeanOfBoundsFunction())
    salaryHierarchy.addInterval(lang.Double(1.5555555555555554), lang.Double(1.7777777777777777), aggregateFunctionBuilder.createArithmeticMeanOfBoundsFunction())
    salaryHierarchy.addInterval(lang.Double(1.7777777777777777), lang.Double(2.0), aggregateFunctionBuilder.createArithmeticMeanOfBoundsFunction())

    salaryHierarchy
  }

  def loadData: Tuple2[Data, Data] = {

    val dataFileOrg: File = new File("/home/alex/qi3/drl_anonymity/src/examples/q_learn_distorted_sets/distorted_set_-1")
    val dataOrg: Data = Data.create(dataFileOrg, Charset.defaultCharset, ',')

    val dataFileDist: File = new File("/home/alex/qi3/drl_anonymity/src/examples/q_learn_distorted_sets/distorted_set_-2")
    val dataDist: Data = Data.create(dataFileDist, Charset.defaultCharset, ',')

    require(dataOrg.getHandle.getNumRows == dataDist.getHandle.getNumRows)
    require(dataOrg.getHandle.getNumColumns == dataDist.getHandle.getNumColumns)

    // define the attribute types
    System.out.println(s"Number of rows ${dataOrg.getHandle.getNumRows}")
    System.out.println(s"Number of cols ${dataOrg.getHandle.getNumColumns}")

    printHandleTop(handle = dataOrg.getHandle, n = 5)
    System.out.println("Done...")

    (dataOrg, dataDist)
  }

  def experiment1: Unit = {

    val data = loadData

    val dataHandleOrg = data._1.getHandle
    val dataHandleDist = data._2.getHandle

    val summaryStatsDist = dataHandleDist.getStatistics().getSummaryStatistics(true)
    val summaryStatsOrg  = dataHandleOrg.getStatistics().getSummaryStatistics(true)
    // getEquivalenceClassStatistics(); //getEquivalenceClassStatistics();

    for((key, value) <- summaryStatsDist){
      println(s"Column: ${key}")
      println("-----------------------Distorted/Original")
      println(s"distinctNumberOfValues ${value.getNumberOfDistinctValuesAsString}/${summaryStatsOrg.get(key).getNumberOfDistinctValuesAsString}")
      println(s"Mode                   ${value.getModeAsString}/${summaryStatsOrg.get(key).getModeAsString}")
      if(value.isMaxAvailable) {
        println(s"Max                  ${value.getMaxAsString}/${summaryStatsOrg.get(key).getMaxAsString}")
        println(s"Min                  ${value.getMinAsString}/${summaryStatsOrg.get(key).getMinAsString}")
      }
    }
  }

  def runKAnonimity: Unit = {

    // load the data
    //val dataFile: File = new File("/home/alex/qi3/drl_anonymity/data/mocksubjects.csv")
    val dataFile: File = new File("/home/alex/qi3/drl_anonymity/data/hierarchies/normalized_salary_mocksubjects.csv")
    val data: Data = Data.create(dataFile, Charset.defaultCharset, ',')

    printHandleTop(handle = data.getHandle, n = 5)

    // set the attribute types if AttributeType.IDENTIFYING_ATTRIBUTE
    // then the attribute will be removed
    data.getDefinition().setAttributeType("preventative_treatment", AttributeType.IDENTIFYING_ATTRIBUTE)
    data.getDefinition().setAttributeType("gender", AttributeType.IDENTIFYING_ATTRIBUTE)
    data.getDefinition().setAttributeType("education", AttributeType.IDENTIFYING_ATTRIBUTE)
    data.getDefinition().setAttributeType("mutation_status", AttributeType.IDENTIFYING_ATTRIBUTE)
    data.getDefinition().setAttributeType("NHSno", AttributeType.IDENTIFYING_ATTRIBUTE)
    data.getDefinition().setAttributeType("given_name", AttributeType.IDENTIFYING_ATTRIBUTE)
    data.getDefinition().setAttributeType("surname", AttributeType.IDENTIFYING_ATTRIBUTE)
    data.getDefinition().setAttributeType("dob", AttributeType.IDENTIFYING_ATTRIBUTE)

    // keep the diagnosis as an insensitive attribute
    data.getDefinition().setAttributeType("diagnosis", AttributeType.INSENSITIVE_ATTRIBUTE)

    // quasi-sensitive attriutes we set the
    // hierarchies
    // the ethnicity hierarchy file
    val ethnicityHierarchyFile: File = new File("/home/alex/qi3/drl_anonymity/data/hierarchies/ethnicity_hierarchy.csv")
    data.getDefinition().setAttributeType("ethnicity", Hierarchy.create(ethnicityHierarchyFile,
      StandardCharsets.UTF_8, ';'))/*AttributeType.QUASI_IDENTIFYING_ATTRIBUTE)*/

    // the salary hierarchy
    //val salaryHierarchyFile: File = new File("/home/alex/qi3/drl_anonymity/data/hierarchies/salary_hierarchy.csv")
    data.getDefinition().setAttributeType("salary", buildSalaryHierarchy) //AttributeType.QUASI_IDENTIFYING_ATTRIBUTE)


    // create the ethnicity hierarchy
    //val ethnicityHierarchy = Hierarchy.create(ethnicityHierarchyFile,
    //  Charset.defaultCharset, ',')

    // create the hierarchies for the ethnicity and
    // salary
    // Create an instance of the anonymizer
    val anonymizer = new ARXAnonymizer
    val config = ARXConfiguration.create
    config.addPrivacyModel(new KAnonymity(5))
    config.setSuppressionLimit(0.02d)


    // anonymize the data using K-anonimity
    val result = anonymizer.anonymize(data, config)

    // Print info
    printResult(result, data)

    // Process results
    System.out.println(" - Transformed data:")
    printHandle(handle = result.getOutput(false))
    System.out.println("Done!")

  }

  // execute Experiment 1
  //experiment1

  //exploreHierarchy
  // run K-anonimity
  runKAnonimity

}
