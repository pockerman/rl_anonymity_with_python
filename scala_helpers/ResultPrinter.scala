package postprocessor

import scala.util.control.Breaks._
import java.text.DecimalFormat
import collection.JavaConverters.* // asScala
import org.deidentifier.arx.{ARXResult, Data, DataHandle}

/**
 * Utility class to print on the console ARXResult instances
 * Adapted from https://github.com/arx-deidentifier/arx/blob/master/src/example/org/deidentifier/arx/examples/Example.java
 */
object ResultPrinter {

  def printResult(result: ARXResult, data: Data): Unit = { // Print time


    val df1 = new DecimalFormat("#####0.00")
    val sTotal = df1.format(result.getTime / 1000d) + "s"
    System.out.println(" - Time needed: " + sTotal)
    // Extract
    val optimum = result.getGlobalOptimum
    val dataDef = data.getDefinition
    val attrs: Set[String] = dataDef.getQuasiIdentifyingAttributes.asScala.toSet[String]

    val qis = attrs.toArray[String]

    if (optimum == null) {
      System.out.println(" - No solution found!")
      return
    }
    // Initialize
    val identifiers = new Array[StringBuffer](qis.size)
    val generalizations = new Array[StringBuffer](qis.size)
    var lengthI = 0
    var lengthG = 0

    for (i <- 0 until qis.size) {
      identifiers(i) = new StringBuffer
      generalizations(i) = new StringBuffer
      identifiers(i).append(qis(i))
      generalizations(i).append(optimum.getGeneralization(qis(i)))
      if (data.getDefinition.isHierarchyAvailable(qis(i))) generalizations(i).append("/").append(data.getDefinition.getHierarchy(qis(i))(0).length - 1)
      lengthI = Math.max(lengthI, identifiers(i).length)
      lengthG = Math.max(lengthG, generalizations(i).length)
    }

    // Padding
    for (i <- 0 until qis.size) {
      while ( {
        identifiers(i).length < lengthI
      }) identifiers(i).append(" ")
      while ( {
        generalizations(i).length < lengthG
      }) generalizations(i).insert(0, " ")
    }
    // Print
    System.out.println(" - Information loss: " + result.getGlobalOptimum.getLowestScore + " / " + result.getGlobalOptimum.getHighestScore)
    System.out.println(" - Optimal generalization")
    for (i <- 0 until qis.size) {
      System.out.println("   * " + identifiers(i) + ": " + generalizations(i))
    }
    System.out.println(" - Statistics")
    System.out.println(result.getOutput(result.getGlobalOptimum, false).getStatistics.getEquivalenceClassStatistics)
  }

  def printHandle(handle: DataHandle): Unit = {

    val transformed = handle.iterator
    while (transformed.hasNext) {
      System.out.print("   ")
      val item = transformed.next
      System.out.println(item.mkString(" "))
    }
  }

  /**
   * Print the n top items in the handle
   */
  def printHandleTop(handle: DataHandle, n: Int): Unit = {

    val transformed = handle.iterator
    var counter = 0

    breakable{
      while (transformed.hasNext) {
        System.out.print("   ")
        val item = transformed.next
        System.out.println(item.mkString(" "))
        counter += 1

        if(counter >= n) break
      }
    }

  }

}
