#include <Grid/Grid.h>

NAMESPACE_BEGIN(Grid);

struct PackRecord
{
    std::string operatorXml, solverXml;
};

struct VecRecord: Serializable
{
    GRID_SERIALIZABLE_CLASS_MEMBERS(VecRecord,
                                    unsigned int, index,
                                    double,       eval);
    VecRecord(void): index(0), eval(0.) {}
};

inline void readHeader(PackRecord &record, ScidacReader &binReader)
{
    std::string recordXml;

    binReader.readLimeObject(recordXml, SCIDAC_FILE_XML);
    XmlReader xmlReader(recordXml, true, "eigenPackPar");
    xmlReader.push();
    xmlReader.readCurrentSubtree(record.operatorXml);
    xmlReader.nextElement();
    xmlReader.readCurrentSubtree(record.solverXml);
}

template <typename T>
void readElement(T &evec, RealD &eval, const unsigned int index,
 ScidacReader &binReader) {
    VecRecord vecRecord;
    bool      cb = false;
    GridBase  *g = evec.Grid();

    binReader.readScidacFieldRecord(evec, vecRecord);

    if (vecRecord.index != index)
    {
        assert(0);
    }
    for (unsigned int mu = 0; mu < g->Dimensions(); ++mu)
    {
        cb = cb or (g->CheckerBoarded(mu) != 0);
    }
    if (cb)
    {
        evec.Checkerboard() = Odd;
    }
    eval = vecRecord.eval;
}

inline void skipElements(ScidacReader &binReader, const unsigned int n)
{
    for (unsigned int i = 0; i < n; ++i)
    {
        binReader.skipScidacFieldRecord();
    }
}

template <typename T>
static void readPack(std::vector<T> &evec, std::vector<RealD> &eval,
 PackRecord &record, const std::string filename, 
 const unsigned int ki, const unsigned int kf,
 bool multiFile) {

    ScidacReader         binReader;

    if (multiFile)
    {
        std::string fullFilename;

        for(int k = ki; k < kf; ++k) 
        {
            std::cout << "Reading element " << k << std::endl;
            fullFilename = filename + "/v" + std::to_string(k) + ".bin";
            binReader.open(fullFilename);
            readHeader(record, binReader);
            readElement(evec[k - ki], eval[k - ki], k, binReader);
            binReader.close();
        }
    } else {
        binReader.open(filename);
        readHeader(record, binReader);
        skipElements(binReader, ki);
        for(int k = ki; k < kf; ++k) 
        {
            std::cout << "Reading element " << k << std::endl;
            readElement(evec[k - ki], eval[k - ki], k, binReader);
        }
        binReader.close();
    }
}

NAMESPACE_END(Grid);

