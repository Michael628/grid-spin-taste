#include <Grid/Grid.h>
#include <set>
#include <regex>

NAMESPACE_BEGIN(Grid);

#define TIME_MOD(t) (((t) + Nt) % Nt)

void makeTimeSeq(std::vector<std::vector<unsigned int>> &timeSeq, 
                 const std::vector<std::set<unsigned int>> &times,
                 std::vector<unsigned int> &current,
                 const unsigned int depth)
{
    if (depth > 0)
    {
        for (auto t: times[times.size() - depth])
        {
            current[times.size() - depth] = t;
            makeTimeSeq(timeSeq, times, current, depth - 1);
        }
    }
    else
    {
        timeSeq.push_back(current);
    }
}

void makeTimeSeq(std::vector<std::vector<unsigned int>> &timeSeq, 
                 const std::vector<std::set<unsigned int>> &times)
{
    std::vector<unsigned int> current(times.size());

    makeTimeSeq(timeSeq, times, current, times.size());
}

std::set<unsigned int> parseTimeRange(const std::string str, const unsigned int nt)
{
    std::regex               rex("([0-9]+)|(([0-9]+)\\.\\.([0-9]+))");
    std::smatch              sm;
    std::vector<std::string> rstr = strToVec<std::string>(str);
    std::set<unsigned int>   tSet;

    for (auto &s: rstr)
    {
        std::regex_match(s, sm, rex);
        if (sm[1].matched)
        {
            unsigned int t;
            
            t = std::stoi(sm[1].str());
            if (t >= nt)
            {
                assert(0);
            }
            tSet.insert(t);
        }
        else if (sm[2].matched)
        {
            unsigned int ta, tb;

            ta = std::stoi(sm[3].str());
            tb = std::stoi(sm[4].str());
            if ((ta >= nt) or (tb >= nt))
            {
                assert(0);
            }
            for (unsigned int ti = ta; ti <= tb; ++ti)
            {
                tSet.insert(ti);
            }
        }
    }

    return tSet;
}
template <typename C, typename MatLeft, typename MatRight>
inline void accTrMul(C &acc, const MatLeft &a, const MatRight &b)
    {
        const int RowMajor = Eigen::RowMajor;
        const int ColMajor = Eigen::ColMajor;
        if ((MatLeft::Options  == RowMajor) and
            (MatRight::Options == ColMajor))
        {
      thread_for(r,a.rows(),
            {
                C tmp;
#ifdef USE_MKL
                dotuRow(tmp, r, a, b);
#else
                tmp = a.row(r).conjugate().dot(b.col(r));
#endif
                thread_critical
                {
                    acc += tmp;
                }
            });
        }
        else
      {
            thread_for(c,a.cols(),
            {
                C tmp;
#ifdef USE_MKL 
                dotuCol(tmp, c, a, b);
#else
                tmp = a.col(c).conjugate().dot(b.row(c));
#endif
                thread_critical
                {
                    acc += tmp;
                }
            });
        }
}

NAMESPACE_END(Grid);
