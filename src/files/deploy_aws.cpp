#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
using namespace std;

void print(std::vector<int> const &input)
{
    for (int i = 0; i < input.size(); i++)
    {
        std::cout << input.at(i);
        if (i!=input.size()-1)
            cout<< ',';
    }
}


std::vector<std::string> split(const std::string& s, char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}


int main(int argc, char* argv[])
{
    vector<int> result;
    int counter = 0;
    int result_temp = 0;
    
    //assuming one argv
    string t1(argv[1]);
    vector<string> temp_str = split(t1, ',');
    vector<string>::iterator pos; 

    for (pos = temp_str.begin(); pos < temp_str.end(); pos++)
    {
        int temp_int;
        istringstream(*pos) >> temp_int;
        
        if (counter == 0)
        {
            result_temp += temp_int;
            counter++;
            continue;
        }
        if (counter == 1)
            result_temp += temp_int;
            result.push_back(result_temp);
            result_temp = 0;
            counter = 0;
    }    
    print(result);
    return 0;
}
