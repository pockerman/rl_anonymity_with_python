/**
 *  Investigate various output quality measures supplied by ARX
 *
 */
package examples.example_3


import base.DefaultConfiguration
import org.deidentifier.arx.Data
import postprocessor.ResultPrinter.printHandleTop

//import scala.collection.JavaConversions._
//import collection.convert.ImplicitConversionsToScala.map AsScala
import collection.JavaConverters.* // asScala
import collection.convert.ImplicitConversions.*
import java.io.File
import java.nio.charset.Charset

object MeasureDataQuality extends App{

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

    val data = loadData

    // create the hierarchies for the ethnicity and
    // salary

  }

  // execute Experiment 1
  experiment1

}
