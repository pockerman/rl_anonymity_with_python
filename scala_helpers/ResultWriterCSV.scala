package postprocessor

import java.io.File
import org.deidentifier.arx.DataHandle


/**
 * Write to CSV file the given DataHandle
 * @param delimiter
 */
class ResultWriterCSV(val delimiter: Char=',') extends ResultWriter {

  override def save(handle: DataHandle, fileName: String): Unit = {
    handle.save(fileName, delimiter)
  }

  override def save(handle: DataHandle, file: File): Unit = {
    handle.save(file)
  }

}
