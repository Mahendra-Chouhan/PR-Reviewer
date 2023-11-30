// Java Program to illustrate
// reading from text file
// as string in Java
package io;
 
import java.nio.file.*;
;
 
public class ReadTextAsString {
 
    public static String readFileAsString(String fileName)
        throws Exception
    {
        String data = "";
        data = new String(
            Files.readAllBytes(Paths.get(fileName)));
        return data;
    }
 
    public static void main(String[] args) throws Exception
    {
        String data = readFileAsString(
            "C:\\Users\\pankaj\\Desktop\\test.java");
        System.out.println(data);
    }
}
