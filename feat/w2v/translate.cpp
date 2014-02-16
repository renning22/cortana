#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <set>
#include <cstdio>
#include <cmath>
#include <utility>

#include <boost/ref.hpp>
#include <boost/tokenizer.hpp>

using namespace std;
using namespace boost;

typedef long long LL;
LL words;
LL size;

vector<float> mat;
vector<string> vocab;
unordered_map<string,LL> wordindex;


bool load(string const & file_name)
{
  FILE * f = fopen(file_name.c_str(), "rb");
  if( !f )
  {
    cout << "File not found" << endl;
    return false;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  
  cout << "words: " << words << endl;
  cout << "size: " << size << endl;

  mat.resize( words*size );
  vocab.resize( words );
  
  for(LL w=0;w<words;w++)
  {
    char buf[2000],ch;
    fscanf(f, "%s%c", buf, &ch);
    double len = 0;
    for (LL a = 0; a < size; a++) fread(&mat[w*size + a], sizeof(float), 1, f);
    len = 0;
    for (LL a = 0; a < size; a++) len += mat[w*size + a] * mat[w*size + a];
    len = sqrt(len);
    for (LL a = 0; a < size; a++) mat[w*size + a] /= len;
    vocab[w] = string(buf);
    wordindex[string(buf)] = w;
  }

  cout << "Loaded" << endl;

  fclose(f);
  return true;
}

double cosine(float * a,float * b)
{
  double s = 0;
  for(LL i=0;i<size;i++)
    s += a[i] * b[i];
  return s;
}

double cosine(string & a,string & b)
{
  LL i1,i2;
  if( !wordindex.count(a) || !wordindex.count(b) )
    return 0;
  i1 = wordindex[a];
  i2 = wordindex[b];
  return cosine((float*)&mat[i1*size],(float*)&mat[i2*size]);
}

vector< pair<string&,float> > nearest(float * vec,LL k=3)
{
  auto myless = [] (pair<string&,float> const & a, pair<string&,float> const & b) -> bool
    {
      return a.second < b.second;
    };

  set< pair<string&,float> , decltype(myless) > hp(myless);
  for(LL w=0;w<words;w++) {
    double s = cosine( &mat[w*size] , vec );
    auto ths = make_pair(boost::ref(vocab[w]),s);
    hp.insert(ths);

    if( hp.size() > k )
      hp.erase( hp.begin() );
  }
  return vector< pair<string&,float> >( hp.begin() , hp.end() );
}

vector< pair<string&,float> > nearest(vector<float> & vec,LL k=3)
{
  return nearest(&*vec.begin(),k);
}

vector< pair<string&,float> > nearest(string const & word,LL k=3)
{
  if( !wordindex.count(word) )
    return {};
  auto ret = nearest(&mat[wordindex[word]*size] , k+1);
  ret.pop_back();
  return ret;
}


void trasnlate(string filename)
{
  
}

int main()
{
  cout << "Hi" << endl;
  load("../../data/news.w2v.bin");

  string word;
  cin >> word;
  auto ret = nearest(word);
  for(auto i=ret.begin();i!=ret.end();i++)
  {
    cout << i->first << " , " << i->second << endl;
  }
  return 0;
}
