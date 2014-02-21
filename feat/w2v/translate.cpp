#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <set>
#include <cstdio>
#include <cmath>
#include <utility>
#include <sstream>
#include <algorithm>
#include <iterator>

#include <boost/ref.hpp>
#include <boost/tokenizer.hpp>

#include <nanoflann.hpp>

using namespace std;
using namespace boost;

typedef long long ll;
ll words;
const ll size = 200;

vector<float> mat;
vector<string> vocab;
unordered_map<string,ll> wordindex;


double distance(float const * a,float const * b)
{
  double s = 0;
  for(ll i=0;i<size;i++)
    s += (a[i]-b[i])*(a[i]-b[i]);
  s = sqrt(s);
  return s;
}

double cosine(float const * a,float const * b)
{
  double s = 0;
  for(ll i=0;i<size;i++)
    s += a[i] * b[i];
  return s;
}

double cosine(string const & a,string const & b)
{
  ll i1,i2;
  if( !wordindex.count(a) || !wordindex.count(b) )
    return -2;
  i1 = wordindex[a];
  i2 = wordindex[b];
  return cosine((float const *)&mat[i1*size],(float const *)&mat[i2*size]);
}

template <typename T>
struct PointCloud
{
  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return size; }

  // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
  inline T kdtree_distance(const T *p1, const size_t idx_p2,size_t size_) const
  {
    return distance(p1,&mat[idx_p2]);
  }

  // Returns the dim'th component of the idx'th point in the class:
  // Since this is inlined and the "dim" argument is typically an immediate value, the
  //  "if/else's" are actually solved at compile time.
  inline T kdtree_get_pt(const size_t idx, int dim) const
  {
    return mat[idx*size+dim];
  }

  // Optional bounding-box computation: return false to default to a standard bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
  //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX &bb) const { return false; }

};

PointCloud<float> cloud;

typedef nanoflann::KDTreeSingleIndexAdaptor<
  nanoflann::L2_Simple_Adaptor<float, PointCloud<float> > ,
  PointCloud<float>,
  size /* dim */
  > my_kd_tree_t;

my_kd_tree_t * index = nullptr;


/////////////////////////////////////


template<class T,class P>
ostream & operator << (ostream & os,pair<T,P> const & t)
{
  os << "(" << t.first << "," << t.second << ")";
  return os;
}

template<class T,class P>
ostream & operator << (ostream & os,tuple<T,P> const & t)
{
  os << "(" << get<0>(t) << "," << get<1>(t) << ")";
  return os;
} 

template<class T>
string join(T const & t,string const & delimiter=",")
{
  stringstream ss;
  for(auto i=t.begin();i!=t.end();++i) {
    if( i != t.begin() )
      ss << delimiter;
    ss << *i;
  }
  return ss.str();
}

bool load(string const & file_name)
{
  FILE * f = fopen(file_name.c_str(), "rb");
  if( !f )
  {
    cout << "file not found" << endl;
    return false;
  }

  ll size_;
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size_);

  assert( size_ == size );
  
  cout << "words: " << words << endl;
  cout << "size: " << size << endl;

  mat.resize( words*size );
  vocab.resize( words );
  
  for(ll w=0;w<words;w++)
  {
    char buf[2000],ch;
    fscanf(f, "%s%c", buf, &ch);
    double len = 0;
    for (ll a = 0; a < size; a++) fread(&mat[w*size + a], sizeof(float), 1, f);
    len = 0;
    for (ll a = 0; a < size; a++) len += mat[w*size + a] * mat[w*size + a];
    len = sqrt(len);
    for (ll a = 0; a < size; a++) mat[w*size + a] /= len;
    vocab[w] = string(buf);
    wordindex[string(buf)] = w;
  }

  cout << "loaded" << endl;

  index = new my_kd_tree_t(size, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );

  index->buildIndex();

  cout << "built kdtree index" << endl;

  fclose(f);
  return true;
}

vector< pair<string&,float> > nearest(float * vec,ll k=3)
{
  auto myless = [] (pair<string&,float> const & a, pair<string&,float> const & b) -> bool
    {
      return a.second < b.second;
    };

  set< pair<string&,float> , decltype(myless) > hp(myless);
  for(ll w=0;w<words;w++) {
    double s = cosine( &mat[w*size] , vec );
    auto ths = make_pair(boost::ref(vocab[w]),s);
    hp.insert(ths);

    if( hp.size() > k )
      hp.erase( hp.begin() );
  }
  return vector< pair<string&,float> >( hp.begin() , hp.end() );
}

vector< pair<string&,float> > nearest(vector<float> & vec,ll k=3)
{
  return nearest(&*vec.begin(),k);
}

vector< pair<string&,float> > nearest(string const & word,ll k=3)
{
  if( !wordindex.count(word) )
    return {};
  auto ret = nearest(&mat[wordindex[word]*size] , k+1);
  ret.pop_back();
  return ret;
}


bool translate_sentence(vector<string> & terms)
{
  const float threshold = 0.5;

  int n = terms.size();
  if( n == 0 )
    return false;
  else if( n == 1 )
    return false;

  vector< tuple<float,int> > pairs;
  for(int i=0;i<n-1;i++){
    pairs.push_back( make_tuple(cosine(terms[i],terms[i+1]),i ) );
    //cout << "(" << get<0>(pairs.back()) << "," << get<1>(pairs.back()) << ") ";
  }

  sort(pairs.begin(),pairs.end());

  string const & str1 = terms[get<1>(pairs.back())];
  string const & str2 = terms[get<1>(pairs.back())+1];
  ll indx1 = wordindex[str1];
  ll indx2 = wordindex[str2];
  vector<float> vec(size);
  for(int i=0;i<size;i++)
    vec[i] = (mat[indx1*size+i] + mat[indx2*size+i])/2;
  
  if( get<0>(pairs.back()) < threshold ) {
    // cout << "Low score pair " << make_tuple(str1,str2) << " , " << get<0>(pairs.back()) << endl;
    return false;
  }

  // KD Tree KNN
  // const int K = 4;
  // size_t ret_index[K];
  // float out_dist_sqr[K];
  // vector<string> near = vector<string>();
  // nanoflann::KNNResultSet<float> resultSet(K);
  // resultSet.init(ret_index, out_dist_sqr);
  // index->findNeighbors(resultSet, &vec[0], nanoflann::SearchParams(10));
  // string replaced;
  // float replaced_score = -1e6;
  // for(auto i=0;i<resultSet.size();i++) {
  //   if( ret_index[i] != indx1 && ret_index[i] != indx2 )
  //     if( out_dist_sqr[i] > replaced_score ) {
  // 	replaced = vocab[ret_index[i]];
  // 	replaced_score = out_dist_sqr[i];
  //     }
  //   near.push_back( vocab[ret_index[i]] );
  // }

  // Naive KNN
  auto const near = nearest(vec,3);
  string replaced;
  float replaced_score = -1e6;
  for(auto i=near.rbegin();i!=near.rend();i++)
    if( i->first != str1 && i->first != str2 ) { replaced = i->first; replaced_score = i->second; break; }

  // if( replaced.empty() ) {
  //   cout << "Error : no nearest neighbour" << endl;
  //   return false;
  // }

  // if( replaced_score <= threshold ) {
  //   cout << "Low score merging phrase " << make_tuple(str1,str2) << " -> " << replaced << " , " << replaced_score << endl;
  //   return false;
  // }


  // cout << endl << "Merge: " << str1 << " , " << str2 << " to ";
  // cout << replaced << " , " << replaced_score << " , " << join(near) << endl;

  vector<string> ret;
  for(int i=0;i<n;i++) {
    if( i == get<1>(pairs.back()) ) {
      ret.push_back( replaced );
      i++;
    }
    else
      ret.push_back( terms[i] );
  }
  terms = ret;
  //cout << join(ret," ") << endl;
  return true;
}

// string translate_sentence(string const & sent)
// {
//   tokenizer<> tok(sent);
//   vector<string> terms(tok.begin(),tok.end());
//   //cout << "tokenized : " << join(terms," ") << endl;
//   vector<string> ret = translate_sentence(terms);
//   return join(ret," ");
// }

string expand_sentence(string const & sent)
{
  vector<string> result;
  tokenizer<> tok(sent);
  vector<string> terms(tok.begin(),tok.end());
  //cout << "tokenized : " << join(terms," ") << endl;

  vector<string> isent;
  for(auto term = terms.begin();term!=terms.end(); term++) {
    ll ret = wordindex.count(*term);
    if( ret )
      isent.push_back(*term);
  }
  //cout << "removed : " << join(terms," ") << endl;

  result.push_back(sent);
  while( translate_sentence(isent) ) {
    string s = join(isent," ");
    result.push_back(s);
  }
  return join(result," - - - ");
}
 
void expand_file(string filename,string outfilename)
{
  ifstream ifs(filename);
  ofstream ofs(outfilename);

  string line;
  ll i = 0;
  while( getline(ifs,line) ) {
    escaped_list_separator<char> sep( '\\', '\t', '"' ) ;
    tokenizer< boost::escaped_list_separator<char> > tok(line,sep);
    vector<string> terms(tok.begin(),tok.end());
    terms[0] = expand_sentence(terms[0]);
    if( terms[0].empty() ) {
      terms[0] = "__empty__";
    }
    ofs << join(terms,"\t") << endl;
    
    i++;
    if( i%100 == 0 )
      cout << i << endl;
  }
}

int main(int argc,char* argv[])
{
  if( argc < 3 )
    return 0;
  
  load("/home/ning/private/w2v/queries_2014-01-15_2014-02-07_zh-CN.filtered.wbr.preprocessed.w2v.bin");

  expand_file(argv[1],argv[2]);

  // string sent;
  // getline(cin,sent);
  // cout << "result: " << endl;
  // cout << expand_sentence(sent) << endl;

  // string word;
  // cin >> word;
  // auto ret = nearest(word);
  // for(auto i=ret.begin();i!=ret.end();i++)
  // {
  //   cout << i->first << " , " << i->second << endl;
  // }
  return 0;
}
