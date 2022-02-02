package postprocessor

import java.io.File
import org.deidentifier.arx.DataHandle


abstract class ResultWriter {

  def save(handle: DataHandle, fileName: String): Unit
  def save(handle: DataHandle, file: File): Unit

}
